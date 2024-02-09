// 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

#![allow(dead_code)]
// https://huggingface.co/facebook/musicgen-small/tree/main
// https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/models/musicgen/modeling_musicgen.py
// TODO: Add an offline mode.
// TODO: Add a KV cache.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod encodec_model;
mod musicgen_model;
mod nn;
use musicgen_model::{GenConfig, MusicgenForConditionalGeneration};
use std::path::Path;
// extern crate candle_core;
use anyhow::{Error as E, Result};
use candle::{DType, IndexOp, Result as Result2, Tensor};
// use candle_core::Result3;
use candle_nn::{
    linear, linear_no_bias, Conv1d, Conv1dConfig, LayerNorm, Linear, Module, VarBuilder,
};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use symphonia::core::conv::IntoSample;

const DTYPE: DType = DType::F32;

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L62
#[derive(Debug, Clone)]
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    span: tracing::Span,
    softmax_span: tracing::Span,
    matmul_span: tracing::Span,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl MultiHeadAttention {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "multi-head-attn");
        let softmax_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-softmax");
        let matmul_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-matmul");
        let query = linear(n_state, n_state, vb.pp("q_proj"))?;
        let value = linear(n_state, n_state, vb.pp("v_proj"))?;
        let key = linear_no_bias(n_state, n_state, vb.pp("k_proj"))?;
        let out = linear(n_state, n_state, vb.pp("out_proj"))?;
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
            span,
            softmax_span,
            matmul_span,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        flush_cache: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let q = self.query.forward(x)?;
        let (k, v) = match xa {
            None => {
                let k = self.key.forward(x)?;
                let v = self.value.forward(x)?;
                (k, v)
            }
            Some(x) => {
                if flush_cache {
                    self.kv_cache = None;
                }
                if let Some((k, v)) = &self.kv_cache {
                    (k.clone(), v.clone())
                } else {
                    let k = self.key.forward(x)?;
                    let v = self.value.forward(x)?;
                    self.kv_cache = Some((k.clone(), v.clone()));
                    (k, v)
                }
            }
        };
        let wv = self.qkv_attention(&q, &k, &v, mask)?;
        let out = self.out.forward(&wv)?;
        Ok(out)
    }

    fn reshape_head(&self, x: &Tensor) -> Result2<Tensor> {
        let (n_batch, n_ctx, n_state) = x.dims3()?;
        let target_dims = &[n_batch, n_ctx, self.n_head, n_state / self.n_head];
        x.reshape(target_dims)?.transpose(1, 2)
    }

    fn qkv_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_, n_ctx, n_state) = q.dims3()?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        let q = (self.reshape_head(q)? * scale)?;
        let k = (self.reshape_head(k)?.transpose(2, 3)? * scale)?;
        let v = self.reshape_head(v)?.contiguous()?;
        let mut qk = {
            let _enter = self.matmul_span.enter();
            q.matmul(&k)?
        };
        if let Some(mask) = mask {
            let mask = mask.i((0..n_ctx, 0..n_ctx))?;
            qk = qk.broadcast_add(&mask)?
        }
        let w = {
            let _enter = self.softmax_span.enter();
            candle_nn::ops::softmax_last_dim(&qk)?
        };
        let wv = {
            let _enter = self.matmul_span.enter();
            w.matmul(&v)?
        }
        .transpose(1, 2)?
        .flatten_from(2)?;
        Ok(wv)
    }
}

#[derive(Debug, Clone)]
struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<(MultiHeadAttention, LayerNorm)>,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    mlp_ln: LayerNorm,
    span: tracing::Span,
}

// impl ResidualAttentionBlock {
//     fn load(n_state: usize, n_head: usize, ca: bool, vb: VarBuilder) -> Result<Self> {
//         let span = tracing::span!(tracing::Level::TRACE, "residual-attn");
//         let attn = MultiHeadAttention::load(n_state, n_head, vb.pp("self_attn"))?;
//         let attn_ln = layer_norm(n_state, vb.pp("self_attn_layer_norm"))?;
//         let cross_attn = if ca {
//             let cross_attn = MultiHeadAttention::load(n_state, n_head, vb.pp("encoder_attn"))?;
//             let cross_attn_ln = layer_norm(n_state, vb.pp("encoder_attn_layer_norm"))?;
//             Some((cross_attn, cross_attn_ln))
//         } else {
//             None
//         };
//         let n_mlp = n_state * 4;
//         let mlp_linear1 = linear(n_state, n_mlp, vb.pp("fc1"))?;
//         let mlp_linear2 = linear(n_mlp, n_state, vb.pp("fc2"))?;
//         let mlp_ln = layer_norm(n_state, vb.pp("final_layer_norm"))?;
//         Ok(Self {
//             attn,
//             attn_ln,
//             cross_attn,
//             mlp_linear1,
//             mlp_linear2,
//             mlp_ln,
//             span,
//         })
//     }

//     fn forward(
//         &mut self,
//         x: &Tensor,
//         xa: Option<&Tensor>,
//         mask: Option<&Tensor>,
//         flush_kv_cache: bool,
//     ) -> Result<()> {
//         let _enter = self.span.enter();
//         let attn = self
//             .attn
//             .forward(&self.attn_ln.forward(x)?, None, mask, flush_kv_cache);
//         let mut x = (x + attn)?;
//         if let Some((attn, ln)) = &mut self.cross_attn {
//             x = (&x + attn.forward(&ln.forward(&x)?, xa, None, flush_kv_cache))?;
//         }
//         let mlp = self.mlp_linear2.forward(
//             &self
//                 .mlp_linear1
//                 .forward(&self.mlp_ln.forward(&x)?)?
//                 .gelu()?,
//         )?;
//         x + mlp
//     }
// }

fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = vb.get((out_channels, in_channels, kernel_size), "weight")?;
    let bias = vb.get(out_channels, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

fn layer_norm(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, 1e-5))
}

// // https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L143
// #[derive(Debug, Clone)]
// pub struct AudioEncoder {
//     conv1: Conv1d,
//     conv2: Conv1d,
//     positional_embedding: Tensor,
//     blocks: Vec<ResidualAttentionBlock>,
//     ln_post: LayerNorm,
//     span: tracing::Span,
//     conv1_span: tracing::Span,
//     conv2_span: tracing::Span,
// }

// impl AudioEncoder {
//     fn load(vb: VarBuilder, cfg: &Config) -> Result2<Self> {
//         let span = tracing::span!(tracing::Level::TRACE, "audio-encoder");
//         let conv1_span = tracing::span!(tracing::Level::TRACE, "conv1");
//         let conv2_span = tracing::span!(tracing::Level::TRACE, "conv2");
//         let n_state = cfg.d_model;
//         let n_head = cfg.encoder_attention_heads;
//         let n_ctx = cfg.max_source_positions;
//         let cfg1 = Conv1dConfig {
//             padding: 1,
//             stride: 1,
//             groups: 1,
//             dilation: 1,
//         };
//         let cfg2 = Conv1dConfig {
//             padding: 1,
//             stride: 2,
//             groups: 1,
//             dilation: 1,
//         };
//         let conv1 = conv1d(cfg.num_mel_bins, n_state, 3, cfg1, vb.pp("conv1"))?;
//         let conv2 = conv1d(n_state, n_state, 3, cfg2, vb.pp("conv2"))?;
//         let positional_embedding = sinusoids(n_ctx, n_state, vb.device())?;
//         let blocks = (0..cfg.encoder_layers)
//             .map(|i| {
//                 ResidualAttentionBlock::load(n_state, n_head, false, vb.pp(&format!("layers.{i}")))
//             })
//             .collect::<Result<Vec<_>>>()?;
//         let ln_post = layer_norm(n_state, vb.pp("layer_norm"))?;
//         Ok(Self {
//             conv1,
//             conv2,
//             positional_embedding,
//             blocks,
//             ln_post,
//             conv1_span,
//             conv2_span,
//             span,
//         })
//     }

//     pub fn forward(&mut self, x: &Tensor, flush_kv_cache: bool) -> Result<Tensor> {
//         let _enter = self.span.enter();
//         let x = {
//             let _enter = self.conv1_span.enter();
//             self.conv1.forward(x)?.gelu()?
//         };
//         let x = {
//             let _enter = self.conv2_span.enter();
//             self.conv2.forward(&x)?.gelu()?
//         };
//         let x = x.transpose(1, 2)?;
//         let (_bsize, seq_len, _hidden) = x.dims3()?;
//         let positional_embedding = self.positional_embedding.narrow(0, 0, seq_len)?;
//         let mut x = x.broadcast_add(&positional_embedding)?;
//         for block in self.blocks.iter_mut() {
//             x = block.forward(&x, None, None, flush_kv_cache)?
//         }
//         let x = self.ln_post.forward(&x)?;
//         Ok(x)
//     }
// }

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The model weight file, in safetensor format.
    #[arg(long)]
    model: Option<String>,

    /// The tokenizer config.
    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(
        long,
        default_value = "90s rock song with loud guitars and heavy drums"
    )]
    prompt: String,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let tokenizer = match args.tokenizer {
        Some(tokenizer) => std::path::PathBuf::from(tokenizer),
        None => Api::new()?
            .model("facebook/musicgen-small".to_string())
            .get("tokenizer.json")?,
    };
    let mut tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;

    let model = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => Api::new()?
            .repo(Repo::with_revision(
                "facebook/musicgen-small".to_string(),
                RepoType::Model,
                "refs/pr/13".to_string(),
            ))
            .get("model.safetensors")?,
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DTYPE, &device)? };
    let config = GenConfig::small();
    let mut model = MusicgenForConditionalGeneration::load(vb, config)?;

    let tokens = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    println!("tokens: {tokens:?}");
    let tokens = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
    println!("{tokens:?}");
    let embeds = model.text_encoder.forward(&tokens)?;
    println!(" ----");
    println!("{embeds}");

    // let model = EncodecModel::new();  // Replace with your actual model initialization
    // let tensor = ;  // Replace with your actual tensor
    // let audio = model.tensor_to_audio(tensor);

    // // MusicgenForConditionalGeneration.audio_encoder();
    // let rr = MusicgenForConditionalGeneration::into_sample(model);
    // println!("{rr:?}");
    // let r = tokenizer.decode(&tokens.flatten(tokens.dims(), embeds.dims()), false);
    println!(" ----");
    let fl = embeds.clone().flatten_all()?.to_vec1::<f32>().unwrap();
    println!("{fl:?}");
    // safetensors::serialize_to
    let path = Path::new("some/path");
    tokenizer.encode(&fl, true);
    // Use the method

    Ok(())
}

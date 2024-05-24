use std::io::{BufReader, Read};

use anyhow::Result;
use tiny_wasm_runtime::{execution::{runtime::Runtime, value::Value, wasi::WasiSnapshotPreview1}, get_args};
use tracing::span;
use tracing_subscriber::{filter, layer::SubscriberExt, util::SubscriberInitExt, Layer};

fn main() -> Result<()> {
    let config = get_args()?;
    
    let stdout_log = tracing_subscriber::fmt::layer().compact()
        .without_time()
        .with_ansi(!config.disable_ansi_color)
        .with_target(false)
    ;
    tracing_subscriber::registry()
    .with(stdout_log.with_filter(filter::LevelFilter::INFO))
    .try_init()?;
    
    let _ = span!(tracing::Level::INFO, "main").enter();

    let wasm = match config.file.as_str() {
        "-" => {
            let mut reader = BufReader::new(std::io::stdin());

            let mut buf = vec![];
            reader.read_to_end(&mut buf)?;
            wat::parse_bytes(&buf)?.to_vec()
        }
        _ => {
            wat::parse_file(config.file)?
        }
    };
    let wasi = WasiSnapshotPreview1::new();
    let mut rt = Runtime::instanciate_with_wasi(wasm, wasi)?;
    
    tracing::info!("call entry point (_start)");
    let value = rt.call("_start", config.args.into_iter().map(Value::I32).collect())?;

    if let Some(value) = value {
        println!("{value}");
    }

    Ok(())
}

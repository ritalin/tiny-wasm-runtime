use std::io::{BufReader, Read};

use anyhow::Result;
use tiny_wasm_runtime::{execution::{runtime::Runtime, wasi::WasiSnapshotPreview1}, get_args};

fn main() -> Result<()> {
    let config = get_args()?;
    
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
    
    rt.call("_start", vec![])?;

    Ok(())
}

use nom::{bytes::complete::tag, number::complete::le_u32, IResult};

const WASM_MAGIC: &str = "\0asm";

#[derive(Debug, PartialEq, Eq)]
pub struct Module {
    pub magic: String,
    pub version: u32,
}

impl Default for Module {
    fn default() -> Self {
        Self { magic: WASM_MAGIC.to_string(), version: 1 }
    }
}

impl Module {
    pub fn new(wasm: &[u8]) -> anyhow::Result<Self> {
        let (_, module) = Module::decode(wasm)
            .map_err(|err| anyhow::anyhow!("failed to parse wasm: {}", err))?;
        Ok(module)
    }

    fn decode(input: &[u8]) -> IResult<&[u8], Module> {
        let (input, _) = tag(WASM_MAGIC.as_bytes())(input)?;
        let (input, version) = le_u32(input)?;

        Ok((input, Module { magic: WASM_MAGIC.to_string(), version }))
    }
}

#[cfg(test)]
mod tests {
    use crate::binary::module::Module;
    use anyhow::Result;

    #[test]
    fn decode_simplest_module() -> Result<()> {
        // プリアンブル飲みのwasm
        let wasm = wat::parse_str("(module)")?;
        let module = Module::new(&wasm)?;
        assert_eq!(module, Module::default());
        Ok(())
    }
}
use crate::binary::section::SectionCode;
use nom::{bytes::complete::{tag, take}, number::complete::{le_u32, le_u8}, sequence::pair, IResult};
use nom_leb128::leb128_u32;
use num_traits::FromPrimitive;

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

        let module = Module { magic: WASM_MAGIC.to_string(), version };

        Ok((input, module))
    }
}

fn decode_section_header(input: &[u8]) -> IResult<&[u8], (SectionCode, u32)> {
    let (input, (code, sz)) = pair(le_u8, leb128_u32)(input)?;
    Ok((
        input, 
        (
            SectionCode::from_u8(code).expect(&format!("Unexpected section code: {code}")),
            sz,
        )
    ))
}

#[cfg(test)]
mod tests {
    use crate::binary::{module::Module, section::SectionCode};
    use anyhow::Result;

    #[test]
    fn decode_simplest_module() -> Result<()> {
        // プリアンブル飲みのwasm
        let wasm = wat::parse_str("(module)")?;
        let module = Module::new(&wasm)?;
        assert_eq!(module, Module::default());
        Ok(())
    }

    #[test]
    fn decode_section_headers() -> Result<()> {
        assert_eq!((SectionCode::Type, 4u32), super::decode_section_header(&[0x01, 0x04])?.1);
        assert_eq!((SectionCode::Function, 2u32), super::decode_section_header(&[0x03, 0x02])?.1);
        assert_eq!((SectionCode::Code, 4u32), super::decode_section_header(&[0x0a, 0x04])?.1);
        Ok(())
    }
}
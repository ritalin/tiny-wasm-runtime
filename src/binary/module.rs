use crate::binary::section::SectionCode;
use nom::{bytes::complete::{tag, take}, number::complete::{le_u32, le_u8}, sequence::pair, IResult};
use nom_leb128::leb128_u32;
use num_traits::FromPrimitive;

use super::{instruction::Instruction, section::Function, types::FuncType};

const WASM_MAGIC: &str = "\0asm";

#[derive(Debug, PartialEq, Eq)]
pub struct Module {
    pub magic: String,
    pub version: u32,
    pub type_section: Option<Vec<FuncType>>,
    pub fn_section: Option<Vec<u32>>,
    pub code_section: Option<Vec<Function>>,
}

impl Default for Module {
    fn default() -> Self {
        Self { magic: WASM_MAGIC.to_string(), version: 1, type_section: None, fn_section: None, code_section: None }
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

        let mut module = Module { magic: WASM_MAGIC.to_string(), version, ..Default::default() };

        let mut remaining = input;
        
        while !remaining.is_empty() {
            match decode_section_header(remaining) {
                Ok((input, (code, sz))) => {
                    let (rest, section_contents) = take(sz)(input)?;

                    match code {
                        SectionCode::Type => {
                            let (_, tys) = decode_type_section(section_contents)?;
                            module.type_section = Some(tys);
                        }
                        SectionCode::Function => {
                            let (_, fns) = decode_function_section(section_contents)?;
                            module.fn_section = Some(fns);
                        }
                        SectionCode::Code => {
                            let (_, code) = decode_code_section(section_contents)?;
                            module.code_section = Some(code);
                        }
                        _ => unreachable!()
                    }
                    remaining = rest;
                }
                Err(err) => return Err(err),
            }
        }

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

fn decode_type_section(input: &[u8]) -> IResult<&[u8], Vec<FuncType>> {
    Ok((input, vec![FuncType::default()]))
}

fn decode_function_section(input: &[u8]) -> IResult<&[u8], Vec<u32>> {
    let mut fns = vec![];
    let (mut input, count) = leb128_u32(input)?;
    
    for _ in 0..count {
        let (rest, i) = leb128_u32(input)?;
        fns.push(i);
        input = rest;
    }
    
    Ok((input, fns))
}

fn decode_code_section(input: &[u8]) -> IResult<&[u8], Vec<Function>> {
    let mut fns = vec![];

    fns.push(Function {
        locals: vec![],
        code: vec![Instruction::End],
    });

    Ok((input, fns))
}

#[cfg(test)]
mod tests {
    use crate::binary::{instruction::Instruction, module::Module, section::{Function, SectionCode}, types::FuncType};
    use anyhow::Result;

    #[test]
    fn decode_simplest_module() -> Result<()> {
        // プリアンブルのみのwasm
        let wasm = wat::parse_str("(module)")?;
        let module = Module::new(&wasm)?;
        assert_eq!(module, Module::default());
        Ok(())
    }

    #[test]
    fn decode_simplest_fn() -> Result<()> {
        let wasm = wat::parse_str("(module (func))")?;
        let module = Module::new(&wasm)?;

        let expected = Module {
            type_section: Some(vec![FuncType::default()]),
            fn_section: Some(vec![0]),
            code_section: Some(vec![Function { locals: vec![], code: vec![Instruction::End] }]),
            ..Default::default()
        };
        assert_eq!(expected, module);
        Ok(())
    }

    #[test]
    fn decode_section_headers() -> Result<()> {
        assert_eq!((SectionCode::Type, 4u32), super::decode_section_header(&[0x01, 0x04])?.1);
        assert_eq!((SectionCode::Function, 2u32), super::decode_section_header(&[0x03, 0x02])?.1);
        assert_eq!((SectionCode::Code, 4u32), super::decode_section_header(&[0x0a, 0x04])?.1);
        Ok(())
    }

    #[test]
    fn decode_function_sections() -> Result<()> {
        let ret = super::decode_function_section(&[0x03, 0, 0x01, 0, 0x0a])?.1;
        assert_eq!(3, ret.len());
        assert_eq!(vec![0, 1, 0], ret);
        Ok(())
    }
}
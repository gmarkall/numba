
import llvmlite.binding as llvm
import numpy as np

def load_ir(i):
    with open(f'ir_{i}_to_parse.ll', 'r') as f:
        return f.read()

irs = [load_ir(i) for i in range(8)]

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

target = llvm.Target.from_default_triple()
target_machine = target.create_target_machine()

mod0 = llvm.parse_assembly(irs[0])
mcjit = llvm.create_mcjit_compiler(mod0, target_machine)

for ir in irs[1:]:
    mod = llvm.parse_assembly(ir)
    print(f"Adding IR for {mod.name}")
    mcjit.add_module(mod)

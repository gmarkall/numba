from llvmlite import binding as ll
from numba.cuda.cudadrv.nvvm import LibDevice

ll.initialize_all_targets()
ll.initialize_all_asmprinters()


def llvm_to_ptx(irs, options):
    #print(options)
    opt = options['opt']
    cpu = options['arch']
    cpu = cpu.replace('compute_', 'sm_')

    target = ll.Target.from_triple('nvptx64-nvidia-cuda')
    tm = target.create_target_machine(opt=opt, cpu=cpu)
    #breakpoint()

    mods = [ll.parse_assembly(ir) for ir in irs]
    main_module, *other_modules = mods

    # XXX: Need to run the nvvm_reflect pass to eliminate unused libdevice
    # functions.
    libdevice = ll.parse_bitcode(LibDevice().get())
    main_module.link_in(libdevice)

    for other in other_modules:
        main_module.link_in(other)

    return tm.emit_assembly(main_module)

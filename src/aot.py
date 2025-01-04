import shutil

import src.config as cfg
import src.utils as utils

from pathlib import Path


def do_precompilation():
    import src.imaging.interface as imaging_interface
    import src.solving.interface as solving_interface

    for module in [imaging_interface, solving_interface]:
        compiled_module_file_exists = any([x for x in cfg.BUILD_DIR.glob(f'{module.MODULE_NAME}*') if x.is_file()])
        if cfg.ENABLE_NUMBA and not cfg.ENABLE_TAICHI:
            if compiled_module_file_exists:
                print(f'Using existing numba Ahead-Of-Time compiled files for module: {module.MODULE_NAME}')
            else:
                print(f'Performing Ahead-Of-Time numba compilation for module: {module.MODULE_NAME}')
                module.cc.compile()
                compiled_module_file = [x for x in Path(module.MODULE_PATH).parent.glob(f'{module.MODULE_NAME}*') if x.is_file()][0]
                utils.mkdir_if_nonexistent(cfg.BUILD_DIR)
                file_dest = cfg.BUILD_DIR / compiled_module_file.name
                file_dest.unlink(missing_ok=True)
                # compiled_module_file.rename(file_dest)
                shutil.move(compiled_module_file, file_dest)


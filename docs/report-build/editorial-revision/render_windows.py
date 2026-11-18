"""Run the bundled renderer with inherited Windows ACLs for temporary folders.

Python 3.13's mode 0700 gives temp folders a restrictive new Windows DACL.
The sandboxed process then cannot enter its own folders.  Use inherited ACLs
instead, within the caller's permitted temporary directory.
"""
import os
import runpy

original_mkdir = os.mkdir


def inherited_mkdir(path, mode=0o777, *, dir_fd=None):
    if os.name == 'nt' and mode == 0o700:
        mode = 0o777
    return original_mkdir(path, mode, dir_fd=dir_fd)


os.mkdir = inherited_mkdir
runpy.run_path(
    'C:/Users/BiKiMC/.cache/codex-runtimes/codex-primary-runtime/plugins/'
    'openai-primary-runtime/plugins/documents/skills/documents/render_docx.py',
    run_name='__main__',
)

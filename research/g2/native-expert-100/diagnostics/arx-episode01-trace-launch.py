import faulthandler
import runpy
import sys
faulthandler.enable()
faulthandler.dump_traceback_later(18, repeat=True)
script = sys.argv.pop(1)
seen = set()
def steps(frame, event, arg):
    if event == 'line' and frame.f_code.co_name == 'physics':
        line = frame.f_lineno
        if line not in seen:
            print('NATIVE_FIRST_LINE', line, flush=True)
            seen.add(line)
    return steps
sys.settrace(steps)
runpy.run_path(script, run_name='__main__')

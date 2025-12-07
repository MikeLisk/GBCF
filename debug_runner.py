import subprocess
import sys

def run():
    try:
        # Run main.py and capture output
        result = subprocess.run(
            [sys.executable, 'main.py', '--dataset', 'sparse_yelp', '--epochs', '2'],
            cwd=r'f:\code\GBNC-main\GBNC-main - 修改版',
            capture_output=True,
            text=True,
            encoding='utf-8', # Force UTF-8 decoding of child process output
            errors='replace'
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Runner Error: {e}")

if __name__ == "__main__":
    run()

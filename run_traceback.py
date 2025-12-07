import sys
import traceback

def run():
    try:
        import main
        sys.argv = ['main.py', '--dataset', 'sparse_yelp', '--epochs', '2']
        main.main()
    except Exception:
        with open('traceback.log', 'w', encoding='utf-8') as f:
            f.write(traceback.format_exc())
        print("Exception occurred and written to traceback.log")

if __name__ == "__main__":
    run()

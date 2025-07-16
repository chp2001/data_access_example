import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, List, Any
import pprint
import json

# show_util.py
# Slightly over-engineered utility to display objects in a script as if they were shown in a Jupyter notebook cell.
# Does not include caching capabilities, but does allow for finding distinct displayed text chunks and plots
# without having to manually manage file names.

distdir = Path("dist")
distdir.mkdir(exist_ok=True)
def show(obj: Any) -> Path:
    """
    Emulate showing an object in a Jupyter notebook cell.
    
    Puts displayed items into sequential files in the `dist/{Path(__file__).stem}` directory.
    
    This allows easy retrieval of displayed items while also keeping the displays of different files separate.
    """
    # Load or create function-specific index dictionary
    index: Dict[str, int] = getattr(show, 'index', {})
    setattr(show, 'index', index)
    # Pull the caller's file path. 
    # This prevents issues if this function is imported from another file.
    callframe = sys._getframe(1)
    filename = callframe.f_code.co_filename
    # Create a directory for the file if it doesn't exist
    stem = Path(filename).stem
    filedir = distdir / stem
    filedir.mkdir(exist_ok=True)
    # Get current sequence number for this file
    seq = index.setdefault(stem, 0)
    # index[stem] += 1 ## Increment when file successfully created
    # Check if a file with the same base name already exists
    # This is important because we will output multiple file types,
    # but the base names should be unique.
    base_name = f"C_{seq}"
    current_files = list(filedir.glob(f"{base_name}.*"))
    if current_files:
        # If a file with the same base name already exists, we intend to overwrite it.
        # Remove the existing file(s) to avoid conflicts.
        for existing_file in current_files:
            existing_file.unlink()
    # Now, depending on the type of object, we will save it in a specific format.
    # Though, for now, we are just checking if it is a reference to matplotlib.pyplot,
    # to facilitate the conversion of `plt.show()` to `show(plt)` in the code.
    if obj is plt:
        # Check that there *is* a current figure to save
        if not plt.get_fignums():
            raise ValueError("No current figure to save. Please create a figure before calling show(plt).")
        # Save the current figure to a file
        fig_path = filedir / f"{base_name}.png"
        # plt.savefig(fig_path)
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)
        index[stem] += 1
        plt.close()
        # print(f"Saved figure to {fig_path}.")
        return fig_path
    else:
        # Try json, then pprint, then str
        def default_tryer(obj: Any) -> str:
            try:
                return pprint.pformat(obj)
            except Exception:
                return str(obj)
        try:
            # Attempt to serialize the object to JSON
            json_str = json.dumps(obj, indent=2, default=default_tryer)
            file_path = filedir / f"{base_name}.json"
            file_path.write_text(json_str, encoding='utf-8')
            index[stem] += 1
            # print(f"Saved object to {file_path} with JSON serialization.")
            return file_path
        except Exception:
            pass
        try:
            # If JSON serialization fails, fall back to pprint
            pp_str = pprint.pformat(obj)
            file_path = filedir / f"{base_name}.txt"
            file_path.write_text(pp_str, encoding='utf-8')
            index[stem] += 1
            # print(f"Saved object to {file_path} with pprint serialization.")
            return file_path
        except Exception:
            # If pprint also fails, fall back to str
            str_repr = str(obj)
            file_path = filedir / f"{base_name}.txt"
            file_path.write_text(str_repr, encoding='utf-8')
            index[stem] += 1
            # print(f"Saved object to {file_path} with str serialization.")
            return file_path
        
def clear_display() -> List[Path]:
    """
    Clear the display by removing all files in the current display directory.
    
    Returns a list of removed file paths.
    """
    callframe = sys._getframe(1)
    filename = callframe.f_code.co_filename
    stem = Path(filename).stem
    filedir = distdir / stem
    if not filedir.exists():
        print(f"No display directory found for {stem}. Nothing to clear.")
        return []
    
    removed_files = list(filedir.glob("*"))
    for file in removed_files:
        file.unlink()
        # print(f"Removed file: {file}")
    
    return removed_files
        
if __name__ == "__main__":
    targetdir = distdir / Path(__file__).stem
    # Example usage
    def nofig_test()->Any:
        try:
            return show(plt)
        except Exception as e:
            return e
    t0 = nofig_test()
    assert isinstance(t0, Exception), f"Expected an exception when no figure is present, but got a {type(t0).__name__}:\n\t{t0}"
    # assert show("3") == distdir / "C_0.json", "Expected the string '3' to be saved in C_0.json"
    # assert show(3) == distdir / "C_1.json", "Expected the integer 3 to be saved in C_1.json"
    t1 = show("3")
    assert t1 == targetdir / "C_0.json", f"Expected the string '3' to be saved in C_0.json, but got {t1}"
    t2 = show(3)
    assert t2 == targetdir / "C_1.json", f"Expected the integer 3 to be saved in C_1.json, but got {t2}"
    def test_plt():
        plt.plot([1, 2, 3], [4, 5, 6])
        return show(plt)
    t3 = test_plt()
    assert t3 == targetdir / "C_2.png", f"Expected the plot to be saved in C_2.png, but got {t3}"
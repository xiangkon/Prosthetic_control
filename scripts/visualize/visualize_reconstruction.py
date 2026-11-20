"""Inspect reconstructed TUM markers, skeletons and joint angles."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.motion_reconstruction.dispatch import select_algorithm


if __name__ == '__main__':
    algorithm, arguments = select_algorithm(sys.argv[1:])
    if algorithm == 'tum_native':
        from src.motion_reconstruction.tum_native.viewer import main
        # Unified --results/--preview map to native trial directory/--save.
        if '--results' in arguments:
            index = arguments.index('--results')
            directory = arguments[index + 1]
            arguments[index:index + 2] = []
            arguments.insert(0, directory)
        arguments = ['--save' if value == '--preview' else value for value in arguments]
    else:
        from src.motion_reconstruction.legacy_g99.viewer import main
    sys.argv = [sys.argv[0], *arguments]
    main()

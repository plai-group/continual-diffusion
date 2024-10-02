import argparse
import os
from pathlib import Path
import sqlite3
from tqdm import tqdm
import zipfile


"""
NOTE: Call from the project root directory.
"""

BALL_STN = "ball_stn"
BALL_NSTN = "ball_nstn"
WMAZE = "windows_maze"
PLAICRAFT = "plaicraft"
SUPPORTED_DATASETS = [BALL_STN, BALL_NSTN, WMAZE, PLAICRAFT]


def find_ball_paths(dataset_path):
    # All files should be zipped
    return [dataset_path / file for file in os.listdir(dataset_path)]


def find_wmaze_paths(dataset_path):
    # All but mp4 files should be zipped
    return [dataset_path / file for file in os.listdir(dataset_path) if ".mp4" not in file]


def find_plaicraft_paths(dataset_path, player_names):
    # All files related to the player video footages should be zipped
    global_db_path = dataset_path / "global_database.db"
    connection = sqlite3.connect(global_db_path)
    cur = connection.cursor()

    sessions = []
    for player_name in player_names:
        cur.execute("""
            SELECT session_id, start_time, frame_count, fps, player_email
            FROM session_metadata
            WHERE player_name=? AND video=1
            ORDER BY start_time ASC
        """, (player_name,))

        player_sessions = cur.fetchall()
        if player_sessions:
            sessions.append((player_name, player_sessions))
    
    video_paths = []
    for player_name, player_sessions in sessions:
        for session in player_sessions:
            session_id, start_time, frame_count, fps, player_email = session
            encoding_path = Path(dataset_path) / player_email / session_id / "encoded_video"
            video_paths.append(encoding_path)

    file_paths = [global_db_path]
    return file_paths + video_paths


def zip_files_and_directories(dataset_path, paths, zip_name):
    """
    Zips the given files and directories into a zip archive with the specified name,
    maintaining the directory structure relative to a common base directory.

    Args:
        paths (list of PosixPath): List of PosixPath objects pointing to files or directories.
        zip_name (str): Name of the resulting zip file.
    """
    # Convert all paths to absolute paths
    abs_paths = [path.absolute() for path in paths]
    # Compute the common base directory
    # base_dir = Path(os.path.dirname(os.path.commonpath([str(path) for path in abs_paths])))
    base_dir = Path(os.path.dirname(dataset_path.absolute()))
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path in tqdm(abs_paths, desc="Adding paths to zip..."):
            if path.is_file():
                # Calculate the relative path and add to zip
                arcname = path.relative_to(base_dir)
                zipf.write(path, arcname)
            elif path.is_dir():
                # Recursively add all files in the directory
                for file in path.rglob('*'):
                    if file.is_file():
                        arcname = file.relative_to(base_dir)
                        zipf.write(file, arcname)


def main(args):
    # Load sqlite database and retrieve sessions that belong to the player.
    # Run bash command that zips all .db files and encoded_video subdirectories.
    dataset = args.dataset
    dataset_path = Path(os.path.join('datasets', dataset))
    save_path = Path(args.save_path)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if dataset == BALL_STN:
        paths = find_ball_paths(dataset_path)
    elif dataset == BALL_NSTN:
        paths = find_ball_paths(dataset_path)
    elif dataset == WMAZE:
        paths = find_wmaze_paths(dataset_path)
    elif dataset == PLAICRAFT:
        paths = find_plaicraft_paths(dataset_path, [args.player_name])
    else:
        raise Exception(f"Unsupported dataset: {dataset}")

    zip_files_and_directories(dataset_path, paths, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--player_name", default="Kyrie", type=str)
    args = parser.parse_args()
    main(args)

import os
import sqlite3
from pathlib import Path
import subprocess

def concatenate_player_videos(dataset_path, player_names, output_dir):
    """
    Concatenate session videos for each specified player into a single MP4 file.

    :param dataset_path: Path to the dataset folder.
    :param player_names: List of player names to process.
    :param output_dir: Directory where the concatenated videos will be saved.
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Path to the global database
    global_db_path = dataset_path / "global_database.db"

    # Connect to the global database
    connection = sqlite3.connect(global_db_path)
    cursor = connection.cursor()

    # Process each player
    for player_name in player_names:
        print(f"Processing player: {player_name}")

        # Retrieve sessions for the player with video modality
        query = """
            SELECT player_name, session_id, start_time, frame_count, fps, player_email 
            FROM session_metadata 
            WHERE video=1 AND player_name=?
        """
        cursor.execute(query, (player_name,))
        sessions = cursor.fetchall()

        if not sessions:
            print(f"No video sessions found for player '{player_name}'.")
            continue

        # Order sessions by start time
        sessions.sort(key=lambda x: x[2])  # x[2] is start_time

        # Collect MP4 file paths
        mp4_files = []
        for session in sessions:
            player_name_db, session_id, start_time, frame_count, fps, player_email = session
            session_dir = dataset_path / player_email / session_id

            # Find the MP4 file in the session directory
            mp4_file = next(session_dir.glob("*.mp4"), None)

            if mp4_file is None:
                print(f"No MP4 file found in session '{session_id}' for player '{player_name}'.")
                continue

            mp4_files.append(mp4_file)

        if not mp4_files:
            print(f"No MP4 files to concatenate for player '{player_name}'.")
            continue

        # Create a temporary list file for ffmpeg
        list_file_path = output_dir / f"{player_name}_file_list.txt"
        with open(list_file_path, 'w') as f:
            for mp4_file in mp4_files:
                # Escape backslashes and single quotes for Windows paths
                file_path = str(mp4_file).replace('\\', '/').replace("'", "\\'")
                f.write(f"file '{file_path}'\n")

        # Define the output file path
        output_file = output_dir / f"{player_name}_concatenated.mp4"

        # Run ffmpeg command to concatenate the videos
        ffmpeg_command = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-f', 'concat',
            '-safe', '0',
            '-i', str(list_file_path),
            '-c', 'copy',
            str(output_file)
        ]

        print(f"Concatenating videos for player '{player_name}' into '{output_file}'...")
        try:
            subprocess.run(ffmpeg_command, check=True)
            print(f"Successfully created '{output_file}'.")
        except subprocess.CalledProcessError as e:
            print(f"Error during concatenation for player '{player_name}': {e}")

        # Remove the temporary list file
        list_file_path.unlink()

    # Close the database connection
    connection.close()

if __name__ == "__main__":
    dataset_path = "/ubc/cs/research/plai-scratch/jason/continual-diffusion/datasets/plaicraft"
    player_names = ["Alex"]  # Replace with your list of player names
    output_dir = "/ubc/cs/research/plai-scratch/jason/plaicraft_alex.mp4"  # Replace with your desired output directory

    concatenate_player_videos(dataset_path, player_names, output_dir)

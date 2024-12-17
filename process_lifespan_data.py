import os
import pandas as pd
from pathlib import Path

# Final Processing Pipeline with Speed Column Removed
def process_dataset_final(file_path, chunk_size=10799, session_length=900, gap_seconds=5.5 * 3600):
    """
    Full pipeline to process the dataset:
    1. Split into chunks.
    2. Duplicate the first frame in each chunk.
    3. Split chunks into 900-frame sessions.
    4. Combine sessions into one dataset.
    5. Add timestamps.
    6. Remove the 'Speed' column if present.
    
    Parameters:
        file_path (str): Path to the input CSV file.
        chunk_size (int): Number of frames in each chunk before restarting.
        session_length (int): Number of frames per session.
        gap_seconds (float): Gap in seconds between sessions.
    
    Returns:
        pd.DataFrame: Fully processed dataset.
    """
    # Step 1: Load the data
    data = pd.read_csv(file_path)

    # Step 2: Split into chunks
    def split_into_chunks(data, chunk_size):
        return [data.iloc[i:i + chunk_size].copy() for i in range(0, len(data), chunk_size)]
    
    chunks = split_into_chunks(data, chunk_size)

    # Step 3: Duplicate the last frame in each chunk
    def duplicate_last_frame(chunks):
        for i, chunk in enumerate(chunks):
            last_frame = chunk.iloc[-1].copy()
            chunks[i] = pd.concat([chunk, pd.DataFrame([last_frame])], ignore_index=True)
        return chunks

    chunks_with_duplicated_frames = duplicate_last_frame(chunks)

    # Step 4: Split chunks into sessions
    def split_chunks_into_sessions(chunks, session_length):
        sessions = []
        for chunk in chunks:
            for i in range(0, len(chunk), session_length):
                session = chunk.iloc[i:i + session_length].copy()
                sessions.append(session)
        return sessions

    sessions = split_chunks_into_sessions(chunks_with_duplicated_frames, session_length)

    # Step 5: Combine sessions into one dataset
    def combine_sessions_with_continuous_frames(sessions):
        combined = pd.concat(sessions, ignore_index=True)
        combined['Frame'] = combined.index  # Make frame numbering continuous
        return combined

    combined_data = combine_sessions_with_continuous_frames(sessions)

    # Step 6: Add timestamps
    def add_timestamps(data, session_length, gap_seconds):
        timestamps = []
        for i in range(len(data)):
            session_index = i // session_length
            frame_within_session = i % session_length
            timestamp = session_index * (session_length * 2 + gap_seconds) + frame_within_session * 2
            timestamps.append(timestamp)
        data['Timestamp'] = timestamps
        return data

    combined_data = add_timestamps(combined_data, session_length, gap_seconds)

    # Step 7: Remove the 'Speed' column if present
    if 'Speed' in combined_data.columns:
        combined_data.drop(columns=['Speed'], inplace=True)

    return combined_data

def process_all_files():
    # Base directory containing all treatment groups
    base_dir = Path('data/Lifespan')
    
    # Create output directory if it doesn't exist
    output_dir = Path('data/Lifespan_processed')
    output_dir.mkdir(exist_ok=True)
    
    # Process each treatment group directory
    for treatment_dir in base_dir.iterdir():
        if treatment_dir.is_dir():
            treatment_name = treatment_dir.name
            print(f"\nProcessing treatment group: {treatment_name}")
            
            # Create treatment-specific output directory
            treatment_output_dir = output_dir / treatment_name
            treatment_output_dir.mkdir(exist_ok=True)
            
            # Process each CSV file in the treatment directory
            for csv_file in treatment_dir.glob('*.csv'):
                print(f"Processing file: {csv_file.name}")
                try:
                    # Process the file
                    processed_data = process_dataset_final(str(csv_file))
                    
                    # Create output filename
                    output_filename = f"processed_{csv_file.name}"
                    output_path = treatment_output_dir / output_filename
                    
                    # Save processed data
                    processed_data.to_csv(output_path, index=False)
                    print(f"Successfully processed and saved: {output_filename}")
                    
                except Exception as e:
                    print(f"Error processing {csv_file.name}: {str(e)}")

if __name__ == "__main__":
    process_all_files() 
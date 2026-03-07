import sys

def renumber_pdb(input_file, output_file=None):
    """
    Renumber residues in a PDB file starting from 1.
    
    Args:
        input_file (str): Path to input PDB file
        output_file (str): Path to output PDB file (if None, prints to stdout)
    """
    if output_file:
        out_handle = open(output_file, 'w')
    else:
        out_handle = sys.stdout
        
    try:
        with open(input_file, 'r') as f:
            residue_counter = 1
            current_residue = None
            new_lines = []
            
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    # Extract residue information (columns 23-26 in PDB format)
                    residue_id = line[22:26].strip()
                    
                    # If this is a new residue, increment counter
                    if residue_id != current_residue:
                        current_residue = residue_id
                        # Format the new residue number (4 characters, right-justified)
                        new_residue_num = f"{residue_counter:>4}"
                        residue_counter += 1
                    else:
                        # Keep the same residue number
                        new_residue_num = f"{residue_counter-1:>4}"
                    
                    # Replace the residue number in the line
                    new_line = line[:22] + new_residue_num + line[26:]
                    new_lines.append(new_line)
                else:
                    # Keep other lines as they are
                    new_lines.append(line)
            
            # Write output
            for line in new_lines:
                out_handle.write(line)
                
    finally:
        if output_file:
            out_handle.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pdb_reres.py <input_pdb> [output_pdb]")
        sys.exit(1)
        
    input_pdb = sys.argv[1]
    output_pdb = sys.argv[2] if len(sys.argv) > 2 else None
    
    renumber_pdb(input_pdb, output_pdb)

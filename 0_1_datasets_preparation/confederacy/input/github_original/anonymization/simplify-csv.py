def process_file(input_filename, output_filename):
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            # Remove newline character
            line = line.rstrip('\n')
            
            # The first token of the line, as separated by "_" or "-" characters.
            first_token = line.split('_', 1)[0].split('-', 1)[0]
            
            # The single character of the line that immediately precedes the last 13.
            char_before_last_13 = '' if len(line) < 14 else line[-14]
            
            # The four characters of the line that immediately precede the last four.
            chars_before_last_4 = '' if len(line) < 8 else line[-8:-4]
            
            # Write output
            outfile.write(f"{first_token}\t{char_before_last_13}\t{chars_before_last_4}\n")

# Example usage
process_file('mapping-grouped.csv', 'mapping-grouped-simple.tsv')
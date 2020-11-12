og_file = 'og_args.py'
output_file = "outfile.txt"

poss_starts = ["parser.add_argument('", 'add_bool_arg(parser,', 'parser.set_defaults(']

var_define = '--'

outfile = ''
with open(og_file, 'r+') as fi:
    for line in fi.readlines():
        for cur_start in poss_starts:
            if cur_start in line:
                var_id = line.split(cur_start)[1].split("',")[0].strip(var_define)
                outfile += '  '+var_id + ': '
                if 'default=' in line:
                    default = line.split('default=')[1].split(",")[0].strip("'")
                    outfile += default
                outfile += '\n'

text_file = open(output_file, "w")
n = text_file.write(outfile)
text_file.close()
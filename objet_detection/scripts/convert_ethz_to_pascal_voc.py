import os
import sys

"""
Anyone reading this. (Probably no one will, but just in case)

This is a very ugly script, but it does the job. 

ETH Zurich have an Thermal Image Dataset and this script is used to convert
their annotation format to the PASCAL VOC format. The goal is to use this
with a pre-trained PASCAL VOC SSD-VGG object detector and try to make an 
awesome thermal infrared object detector. Yay !
"""

path = sys.argv[1]  # /path/to/dir
width = 324
height = 256
lbl_type = None

person_class = ['human', 'human_child', 'person']
horse_class = ['horse']
cat_class = ['cat']

for dirr in os.listdir(path):
    try:
        with open(os.path.join(path, dirr, 'annotation', dirr + '.txt')) as infile:
            annotations = {}
            for line in infile:
                if 'lbl' in line:
                    sublines = line.split(' ')
                    for subline in sublines:
                        if '=' in subline:
                            subline = subline.split('=')
                            if len(subline) % 2 == 0:
                                i = 0
                                while i < len(subline):
                                    key = annotations.get(subline[i].replace(' ', ''), [])
                                    key.append(subline[i + 1])
                                    annotations[subline[i].replace(' ', '')] = key
                                    i += 2
                elif '=' in line:
                    line = line.split('=')
                    if line[0] == 'nFrame':
                        annotations['nFrame'] = line[1].split(' ')[0]
                    elif len(line) % 2 == 0:
                        i = 0
                        while i < len(line):
                            key = annotations.get(line[i].replace(' ', ''), [])
                            key.append(line[i+1])
                            annotations[line[i].replace(' ', '')] = key
                            i += 2

            pos = annotations.get('pos', None)
            posv = annotations.get('posv', None)
            occl = annotations.get('occl', None)
            lock = annotations.get('lock', None)

            if all([pos, posv, occl, lock]):

                processed_pos = []
                # pre-processing
                for val in pos:
                    processed_pos.append([])
                    val = val.replace('[', '').replace(']', '')
                    groups = val.split(';')
                    for group in groups:
                        coords = group.split(' ')
                        if len(coords[1:]) == 4:
                            processed_pos[-1].append(coords[1:])

                processed_posv = []
                # pre-processing
                for val in posv:
                    processed_posv.append([])
                    val = val.replace('[', '').replace(']', '')
                    groups = val.split(';')
                    for group in groups:
                        coords = group.split(' ')
                        if len(coords[1:]) == 4:
                            processed_posv[-1].append(coords[1:])

                processed_occl = []
                for val in occl:
                    val = val.replace('[', '').replace(']', '')
                    processed_occl.append([entry for entry in val.split(' ')])

                processed_lock = []
                for val in lock:
                    val = val.replace('[', '').replace(']', '')
                    processed_lock.append([entry for entry in val.split(' ')])

                obj_cnt = len(processed_pos)
                bbox_indexes = {}
                for i in range(obj_cnt):
                    bbox_indexes[str(i)] = 0

                voc_labels = {}

                for i in range(int(annotations['nFrame'])):
                    voc_labels[str(i)] = []

                for i in range(len(processed_occl)):

                    start = int(annotations['str'][i])
                    end = int(annotations['end'][i])

                    for j in range(int(annotations['nFrame'])):
                        if start <= j and j < end:
                            voc_labels[str(j)].append(processed_pos[i][j-start])

                try:
                    os.mkdir(os.path.join(path, dirr, 'Annotations'))
                except FileExistsError:
                    pass

                for i in range(int(annotations['nFrame'])):
                    try:
                        os.rename(os.path.join(path, dirr, 'JPEGImages', str(i+1) + '.png'), os.path.join(path, dirr, 'JPEGImages', dirr + '_' + str(i+1) + '.png'))
                    except FileNotFoundError:
                        pass
                    with open(os.path.join(path, dirr, 'Annotations', dirr +'_' +str(i+1) + '.xml'), 'w') as outfile:
                        outfile.write('<annotation><folder>JPEGImages</folder><filename>{0}_{1}.png</filename>'.format(dirr, str(i)))
                        outfile.write('<path>/home/karl/workspace/data/{0}/JPEGImages/{1}_{2}.png</path>'.format(dirr, dirr, str(i)))
                        outfile.write('<source><database>Unknown</database></source>')
                        outfile.write('<size><width>324</width><height>256</height><depth>1</depth></size>')
                        outfile.write('<segmented>0</segmented>')

                        labels = voc_labels[str(i)]
                        for j in range(len(labels)):
                            outfile.write('<object><name>person</name><pose>Unspecified</pose>')
                            outfile.write('<truncated>0</truncated><difficult>0</difficult>')
                            label = labels[j]
                            outfile.write('<bndbox>')
                            outfile.write('<xmin>{0}</xmin><ymin>{1}</ymin><xmax>{2}</xmax><ymax>{3}</ymax>'.format(
                                int(float(label[0])),
                                int(float(label[1])),
                                int(float(label[0])) + int(float(label[2])),
                                int(float(label[1])) + int(float(label[3]))
                            ))
                            outfile.write('</bndbox>')
                            outfile.write('</object>')

                        outfile.write('</annotation>')

    except Exception as e:
        print(e)

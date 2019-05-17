

def _decode_fpath(fpath, label="ufavratev3"):
    fname = os.path.basename(fpath)
    #label_str = re.findall(re_exp, fname)[0][1:-1]
    label_str = fname

    labelobj = re.search("\[%s_.*?\]"%label, label_str)
    if labelobj is None:
        try:
            label = label_str.split("_")[-2]
            label = float(label)
        except Exception:
            raise ValueError("Fail to extract label from fname")

    else:
        label = labelobj.group()
        label = label[1:-1].split("_")[-1]
        label = float(label)
    try:
        songid = int(label_str.split("[")[0])
    except Exception as e:
        songid = int(label_str.split("_")[0])
    return songid,label

                                                                                                                   1,1           Top

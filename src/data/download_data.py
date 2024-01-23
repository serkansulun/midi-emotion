import gdown

# a file
# url = "https://drive.usercontent.google.com/download?id=1tZKMhYazSWapFTUt7H6abHSo-QKH9ATC&authuser=0"
# url = 'https://drive.usercontent.google.com/open?id=1tZKMhYazSWapFTUt7H6abHSo-QKH9ATC&authuser=0'
id = '1tZKMhYazSWapFTUt7H6abHSo-QKH9ATC'
output = "/nas-ctm01/datasets/public/lpd/lpd_5_full.tar.gz"
gdown.download(id=id, output=output, quiet=False)
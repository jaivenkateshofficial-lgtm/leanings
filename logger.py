import logging

logging.basicConfig(filename="jai.txt",level=logging.DEBUG,format='%(asctime)s-%(name)s %(levelname)s %(message)s',datefmt="%Y-%m-%d %H:%M:%S")
logging.info("This just info for you")

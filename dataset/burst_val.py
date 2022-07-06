from dataset.burstsr_dataset import BurstSRDataset
from data import processing, sampler

def get_burstsr_val_set():
    """ Get the BurstSR validation dataset """
    burstsr_dataset = BurstSRDataset(split='val', initialize=True)
    processing_fn = processing.BurstSRProcessing(transform=None, random_flip=False,
                                                 substract_black_level=True,
                                                 crop_sz=80)
    # Train sampler and loader
    dataset = sampler.IndexedBurst(burstsr_dataset, burst_size=14, processing=processing_fn)
    return dataset

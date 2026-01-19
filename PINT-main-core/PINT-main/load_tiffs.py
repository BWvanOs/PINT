from pathlib import Path
import tifffile
import numpy as np

"""
This module will take in multistack OME.TIFF exported from MCD veiwer (32bit, multistack) files and returns an array of 
Note that this is only compatible with MCD viewer output, or other outputs that use "Pagename" as a tag for data pages.
In the future this will be update to also include normal stacked TIFFs if needs arise.
If this module breaks, it's probably because the OME.TIFF data scructure changed. See _channel_names_from_page_tags module
for more information on this!
"""


def _channel_names_from_page_tags(tif: tifffile.TiffFile):
    """
    This module tries to load the channel names either from the PageNames or pagetags (285, tiffstandard)
    Fallback is _channel_names_from_ome_xml
    Called by the main loading function load_tiffs_raw
    """
    names = []
    has_any = False
    ##looping through the pages and loading them with their name.
    for i, page in enumerate(tif.pages):
        ##This only looks for pages in the ome.tiff that are labeled Pagename. These pages contain the names of the actual tiff that hold the data
        ##If this module fails it's probably because something was changed in the output from MCD viewer
        ##The tag is the PageMame (should be readable by humans, eg a word) or just he pagetag (which is 285 tiff file specification)
        tag = page.tags.get("PageName") or page.tags.get(285)
        ##If can't find name, leave open and set the channel number later
        name = None
        ##Continues with this only if a tag was found. In theory this should not happen when exported via MCD viewer.
        if tag is not None:
            val = tag.value
            if isinstance(val, bytes):
                ##I added this part in version 0.2 because there was an issue with older MCD export seemingly encoding the PagNames as value
                val = val.decode("utf-8", "ignore")
            #Strip spaces for a more coherent name.  
            name = str(val).strip()
        #Build list of names
        if name:
            has_any = True
            names.append(name)
        else:
            names.append(None)
    if has_any:
        #If it found any names, it will return those. Missing channel names are filled with their channel nr -> Channel10, Channel 12 etc
        return [n if n else f"Channel{i+1}" for i, n in enumerate(names)]
    ##If no names found it will not return any names.
    return None

def _channel_names_from_ome_xml(tif: tifffile.TiffFile):
    """
    This is a fallback method for if _channel_names_from_page_tags doesnt return anything usefell and tag-based naming fails. 
    This probably means the data is in a different format.
    It will try to grab the names from the XML metadata of the OME.TIFF
    """
    try:
        #Grab the metadata from the ome file
        ome_xml = tif.ome_metadata
        if not ome_xml:
            ##If it cant find the ome data it probably means it was never there. return none
            return None
        from ome_types import from_xml  #Loads ome_types if the fallback function is used. 
        ome = from_xml(ome_xml)
        ##Go into the first image and read channels from pixel elements. If there is no channel list there
        ##20251107 I'm not sure his actually works when i'm looking at the XML data structure of the MCD export
        chans = getattr(ome.images[0].pixels, "channels", None)
        if not chans:
            #No list? Return none
            return None
        ##Empty channel list created
        out = []
        ##For each channel names try to get channel name, if fails, try channel ID, if fails fall back to channel1, channel2 etc.
        for i, ch in enumerate(chans):
            nm = getattr(ch, "name", None) or getattr(ch, "id", None) or f"Channel{i+1}"
            if isinstance(nm, str):
                nm = nm.strip()
            out.append(nm)
        return out or None
    except Exception as e:
        ##If also fails print the error
        print(f"[OME] Could not parse channel names from OME-XML: {e}")
        return None

def load_tiffs_raw(folderPath: str):
    """
    This module loads the IMC OME-TIFFs as raw multi-page TIFFs.
    Each page/frame = one channel. So each datafile is loaded as an matrix
    Returns:
        imagesDict: {sampleName: imageArray [C, Y, X]}
        channelNamesDict: {sampleName: [channel names]}
    So image array and name list is seperate.
    """
    ##Prepare outputs. Folderpath is entered by used
    folderPath = Path(folderPath)
    ##List of tifffiles that needs to be loaded
    tiffFiles = sorted(folderPath.glob("*.ome.tif*"))

    #Image array and channel lists
    imagesDict = {}
    channelNamesDict = {}

    ##For each tiff file load the images
    for f in tiffFiles:
        with tifffile.TiffFile(f) as tif:
            # Stack each page as a matrix into a stack [C, Y, X]. So it's a stack matrix with all images (C) stacked 
            pages = [page.asarray() for page in tif.pages]
            imageArray = np.stack(pages, axis=0)

            # Try channel names from TIFF page tags first, if that fails, from OME-XML, if not dummy list will be used
            ch_names = _channel_names_from_page_tags(tif)
            if not ch_names:
                ch_names = _channel_names_from_ome_xml(tif)

        #remove file extension from name
        sampleName = f.stem.replace(".ome", "")
        ##Stores the samples in the dictionary of samples under the current samplename
        imagesDict[sampleName] = imageArray

        #If there are still no names to be found create channel names channel"n", so channel1, channel2 etc
        nC = imageArray.shape[0]
        if not ch_names:
            ch_names = [f"Channel{i+1}" for i in range(nC)]
        else:
            #Ensure length of the channelist matches number of planes (if not this means something went wrong with loading pages or setting names)
            #If too short, pad the list, if too long truncate.
            ##Note, future version should handle this better, in theory this should not be possible
            if len(ch_names) < nC:
                ch_names = ch_names + [f"Channel{i+1}" for i in range(len(ch_names), nC)]
            elif len(ch_names) > nC:
                ch_names = ch_names[:nC]

        channelNamesDict[sampleName] = ch_names
        preview = ", ".join(ch_names[:5]) + ("..." if nC > 5 else "")

        ##Print out into the console what was loaded
        print(f"Loaded {f.name} → shape {imageArray.shape}, channels=[{preview}]")

    #Return the images an channelname dictionaries to the main function that called it.
    return imagesDict, channelNamesDict

#Optional CLI test Not really functional as a ttest at the moment
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python load_tiffs.py /path/to/ome-tiffs")
        raise SystemExit(1)
    folder = sys.argv[1]
    imagesDict, channelNamesDict = load_tiffs_raw(folder)
    firstSample = next(iter(imagesDict))
    print("First sample shape:", imagesDict[firstSample].shape)
    print("Channel names:", channelNamesDict[firstSample])

from models.cnn_module import TwoConvBlock, TwoConvSEBlock, TwoConvResidualBlock

ALL_CONVBLOCK_TYPES = dict(
    TwoConvBlock=TwoConvBlock,
    TwoConvSEBlock=TwoConvSEBlock,
    TwoConvResidualBlock=TwoConvResidualBlock
)

def get_block(block_name: str, kwargs):
    block = ALL_CONVBLOCK_TYPES.get(block_name, None)
    assert block is not None, 'block type do not exsist'
    return block(**kwargs)


import NetworkConfig

config_name = input('Name of base config: ')
config = NetworkConfig.NetworkConfig()
config.read_config_from_file(config_name)
print('Make desired changes to config')
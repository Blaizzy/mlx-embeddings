require "json"

module Cask
  class Config < DelegateClass(Hash)
    DEFAULT_DIRS = {
      appdir:               "/Applications",
      prefpanedir:          "~/Library/PreferencePanes",
      qlplugindir:          "~/Library/QuickLook",
      dictionarydir:        "~/Library/Dictionaries",
      fontdir:              "~/Library/Fonts",
      colorpickerdir:       "~/Library/ColorPickers",
      servicedir:           "~/Library/Services",
      input_methoddir:      "~/Library/Input Methods",
      internet_plugindir:   "~/Library/Internet Plug-Ins",
      audio_unit_plugindir: "~/Library/Audio/Plug-Ins/Components",
      vst_plugindir:        "~/Library/Audio/Plug-Ins/VST",
      vst3_plugindir:       "~/Library/Audio/Plug-Ins/VST3",
      screen_saverdir:      "~/Library/Screen Savers",
    }.freeze

    def self.global
      @global ||= new
    end

    def self.for_cask(cask)
      if cask.config_path.exist?
        from_file(cask.config_path)
      else
        global
      end
    end

    def self.from_file(path)
      config = begin
        JSON.parse(File.read(path))
      rescue JSON::ParserError => e
        raise e, "Cannot parse #{path}: #{e}", e.backtrace
      end

      new(Hash[config.map { |k, v| [k.to_sym, v] }])
    end

    def initialize(**dirs)
      super(Hash[DEFAULT_DIRS.map { |(k, v)| [k, Pathname(dirs.fetch(k, v)).expand_path] }])
    end

    def binarydir
      @binarydir ||= HOMEBREW_PREFIX/"bin"
    end

    DEFAULT_DIRS.keys.each do |dir|
      define_method(dir) do
        self[dir]
      end

      define_method(:"#{dir}=") do |path|
        self[dir] = Pathname(path).expand_path
      end
    end

    def write(path)
      path.atomic_write(to_json)
    end
  end
end

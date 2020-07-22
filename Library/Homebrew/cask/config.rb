# frozen_string_literal: true

require "json"

require "lazy_object"
require "locale"

require "extend/hash_validator"
using HashValidator

module Cask
  class Config
    DEFAULT_DIRS = {
      appdir:               "/Applications",
      prefpanedir:          "~/Library/PreferencePanes",
      qlplugindir:          "~/Library/QuickLook",
      mdimporterdir:        "~/Library/Spotlight",
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

    def self.defaults
      {
        languages: LazyObject.new { MacOS.languages },
      }.merge(DEFAULT_DIRS).freeze
    end

    def self.global
      @global ||= new
    end

    def self.clear
      @global = nil
    end

    def self.for_cask(cask)
      if cask.config_path.exist?
        from_json(File.read(cask.config_path))
      else
        global
      end
    end

    def self.from_json(json)
      config = begin
        JSON.parse(json)
      rescue JSON::ParserError => e
        raise e, "Cannot parse #{path}: #{e}", e.backtrace
      end

      new(
        default:  config.fetch("default",  {}),
        env:      config.fetch("env",      {}),
        explicit: config.fetch("explicit", {}),
      )
    end

    def self.canonicalize(config)
      config.map do |k, v|
        key = k.to_sym

        if DEFAULT_DIRS.key?(key)
          [key, Pathname(v).expand_path]
        else
          [key, v]
        end
      end.to_h
    end

    attr_accessor :explicit

    def initialize(default: nil, env: nil, explicit: {})
      @default = self.class.canonicalize(self.class.defaults.merge(default)) if default
      @env = self.class.canonicalize(env) if env
      @explicit = self.class.canonicalize(explicit)

      @env&.assert_valid_keys!(*self.class.defaults.keys)
      @explicit.assert_valid_keys!(*self.class.defaults.keys)
    end

    def default
      @default ||= self.class.canonicalize(self.class.defaults)
    end

    def env
      @env ||= self.class.canonicalize(
        Shellwords.shellsplit(ENV.fetch("HOMEBREW_CASK_OPTS", ""))
                  .select { |arg| arg.include?("=") }
                  .map { |arg| arg.split("=", 2) }
                  .map do |(flag, value)|
                    key = flag.sub(/^--/, "")

                    if key == "language"
                      key = "languages"
                      value = value.split(",")
                    end

                    [key, value]
                  end,
      )
    end

    def binarydir
      @binarydir ||= HOMEBREW_PREFIX/"bin"
    end

    def manpagedir
      @manpagedir ||= HOMEBREW_PREFIX/"share/man"
    end

    def languages
      [
        *explicit[:languages],
        *env[:languages],
        *default[:languages],
      ].uniq.select do |lang|
        # Ensure all languages are valid.
        Locale.parse(lang)
        true
      rescue Locale::ParserError
        false
      end
    end

    def languages=(languages)
      explicit[:languages] = languages
    end

    DEFAULT_DIRS.each_key do |dir|
      define_method(dir) do
        explicit.fetch(dir, env.fetch(dir, default.fetch(dir)))
      end

      define_method(:"#{dir}=") do |path|
        explicit[dir] = Pathname(path).expand_path
      end
    end

    def merge(other)
      self.class.new(explicit: other.explicit.merge(explicit))
    end

    def to_json(*args)
      {
        default:  default,
        env:      env,
        explicit: explicit,
      }.to_json(*args)
    end
  end
end

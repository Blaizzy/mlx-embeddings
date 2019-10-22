# frozen_string_literal: true

require "json"

require "extend/hash_validator"
using HashValidator

module Cask
  class Config
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

    def self.clear
      @global = nil
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
      @default = self.class.canonicalize(default) if default
      @env = self.class.canonicalize(env) if env
      @explicit = self.class.canonicalize(explicit)

      @env&.assert_valid_keys!(*DEFAULT_DIRS.keys)
      @explicit.assert_valid_keys!(*DEFAULT_DIRS.keys)
    end

    def default
      @default ||= self.class.canonicalize(DEFAULT_DIRS)
    end

    def env
      @env ||= self.class.canonicalize(
        Shellwords.shellsplit(ENV.fetch("HOMEBREW_CASK_OPTS", ""))
                  .select { |arg| arg.include?("=") }
                  .map { |arg| arg.split("=", 2) }
                  .map { |(flag, value)| [flag.sub(/^\-\-/, ""), value] },
      )
    end

    def binarydir
      @binarydir ||= HOMEBREW_PREFIX/"bin"
    end

    def manpagedir
      @manpagedir ||= HOMEBREW_PREFIX/"share/man"
    end

    DEFAULT_DIRS.keys.each do |dir|
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

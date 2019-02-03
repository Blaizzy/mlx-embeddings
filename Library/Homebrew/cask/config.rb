require "json"

require "extend/hash_validator"
using HashValidator

module Cask
  class Config
    DEFAULT_DIRS = {
      appdir:               Pathname("/Applications").expand_path,
      prefpanedir:          Pathname("~/Library/PreferencePanes").expand_path,
      qlplugindir:          Pathname("~/Library/QuickLook").expand_path,
      dictionarydir:        Pathname("~/Library/Dictionaries").expand_path,
      fontdir:              Pathname("~/Library/Fonts").expand_path,
      colorpickerdir:       Pathname("~/Library/ColorPickers").expand_path,
      servicedir:           Pathname("~/Library/Services").expand_path,
      input_methoddir:      Pathname("~/Library/Input Methods").expand_path,
      internet_plugindir:   Pathname("~/Library/Internet Plug-Ins").expand_path,
      audio_unit_plugindir: Pathname("~/Library/Audio/Plug-Ins/Components").expand_path,
      vst_plugindir:        Pathname("~/Library/Audio/Plug-Ins/VST").expand_path,
      vst3_plugindir:       Pathname("~/Library/Audio/Plug-Ins/VST3").expand_path,
      screen_saverdir:      Pathname("~/Library/Screen Savers").expand_path,
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
        default:  config.fetch("default",  {}).map { |k, v| [k.to_sym, Pathname(v).expand_path] }.to_h,
        env:      config.fetch("env",      {}).map { |k, v| [k.to_sym, Pathname(v).expand_path] }.to_h,
        explicit: config.fetch("explicit", {}).map { |k, v| [k.to_sym, Pathname(v).expand_path] }.to_h,
      )
    end

    attr_accessor :explicit

    def initialize(default: nil, env: nil, explicit: {})
      env&.assert_valid_keys!(*DEFAULT_DIRS.keys)
      explicit.assert_valid_keys!(*DEFAULT_DIRS.keys)

      @default = default
      @env = env
      @explicit = explicit.map { |(k, v)| [k.to_sym, Pathname(v).expand_path] }.to_h
    end

    def default
      @default ||= DEFAULT_DIRS
    end

    def env
      @env ||= Shellwords.shellsplit(ENV.fetch("HOMEBREW_CASK_OPTS", ""))
                         .map { |arg| arg.split("=", 2) }
                         .map { |(flag, value)| [flag.sub(/^\-\-/, "").to_sym, Pathname(value).expand_path] }
                         .to_h
    end

    def binarydir
      @binarydir ||= HOMEBREW_PREFIX/"bin"
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

    def write(path)
      path.atomic_write(to_json)
    end
  end
end

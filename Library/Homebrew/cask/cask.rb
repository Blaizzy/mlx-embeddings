# frozen_string_literal: true

require "cask/cask_loader"
require "cask/config"
require "cask/dsl"
require "cask/metadata"
require "searchable"

module Cask
  class Cask
    extend Enumerable
    extend Forwardable
    extend Searchable
    include Metadata

    attr_reader :token, :sourcefile_path, :config

    def self.each
      return to_enum unless block_given?

      Tap.flat_map(&:cask_files).each do |f|
        yield CaskLoader::FromTapPathLoader.new(f).load
      rescue CaskUnreadableError => e
        opoo e.message
      end
    end

    def tap
      return super if block_given? # Object#tap

      @tap
    end

    def initialize(token, sourcefile_path: nil, tap: nil, &block)
      @token = token
      @sourcefile_path = sourcefile_path
      @tap = tap
      @block = block
      self.config = Config.for_cask(self)
    end

    def config=(config)
      @config = config

      @dsl = DSL.new(self)
      return unless @block

      @dsl.instance_eval(&@block)
      @dsl.language_eval
    end

    DSL::DSL_METHODS.each do |method_name|
      define_method(method_name) { |&block| @dsl.send(method_name, &block) }
    end

    def timestamped_versions
      Pathname.glob(metadata_timestamped_path(version: "*", timestamp: "*"))
              .map { |p| p.relative_path_from(p.parent.parent) }
              .sort_by(&:basename) # sort by timestamp
              .map { |p| p.split.map(&:to_s) }
    end

    def versions
      timestamped_versions.map(&:first)
                          .reverse
                          .uniq
                          .reverse
    end

    def full_name
      return token if tap.nil?
      return token if tap.user == "Homebrew"

      "#{tap.name}/#{token}"
    end

    def installed?
      !versions.empty?
    end

    def install_time
      _, time = timestamped_versions.last
      return unless time

      Time.strptime(time, Metadata::TIMESTAMP_FORMAT)
    end

    def installed_caskfile
      installed_version = timestamped_versions.last
      metadata_master_container_path.join(*installed_version, "Casks", "#{token}.rb")
    end

    def config_path
      metadata_master_container_path/"config.json"
    end

    def caskroom_path
      @caskroom_path ||= Caskroom.path.join(token)
    end

    def outdated?(greedy = false)
      !outdated_versions(greedy).empty?
    end

    def outdated_versions(greedy = false)
      # special case: tap version is not available
      return [] if version.nil?

      if greedy
        return versions if version.latest?
      elsif auto_updates
        return []
      end

      installed = versions
      current   = installed.last

      # not outdated unless there is a different version on tap
      return [] if current == version

      # collect all installed versions that are different than tap version and return them
      installed.reject { |v| v == version }
    end

    def to_s
      @token
    end

    def hash
      token.hash
    end

    def eql?(other)
      token == other.token
    end
    alias == eql?

    def to_h
      {
        "token"          => token,
        "name"           => name,
        "homepage"       => homepage,
        "url"            => url,
        "appcast"        => appcast,
        "version"        => version,
        "sha256"         => sha256,
        "artifacts"      => artifacts.map(&method(:to_h_gsubs)),
        "caveats"        => (to_h_string_gsubs(caveats) unless caveats.empty?),
        "depends_on"     => depends_on,
        "conflicts_with" => conflicts_with,
        "container"      => container,
        "auto_updates"   => auto_updates,
      }
    end

    private

    def to_h_string_gsubs(string)
      string.to_s
            .gsub(ENV["HOME"], "$HOME")
            .gsub(HOMEBREW_PREFIX, "$(brew --prefix)")
    end

    def to_h_array_gsubs(array)
      array.to_a.map do |value|
        to_h_gsubs(value)
      end
    end

    def to_h_hash_gsubs(hash)
      hash.to_h.each_with_object({}) do |(key, value), h|
        h[key] = to_h_gsubs(value)
      end
    rescue TypeError
      to_h_array_gsubs(hash)
    end

    def to_h_gsubs(value)
      if value.respond_to? :to_h
        to_h_hash_gsubs(value)
      elsif value.respond_to? :to_a
        to_h_array_gsubs(value)
      else
        to_h_string_gsubs(value)
      end
    end
  end
end

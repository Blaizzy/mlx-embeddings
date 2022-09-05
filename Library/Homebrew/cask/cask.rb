# typed: false
# frozen_string_literal: true

require "cask/cask_loader"
require "cask/config"
require "cask/dsl"
require "cask/metadata"
require "searchable"
require "utils/bottles"

module Cask
  # An instance of a cask.
  #
  # @api private
  class Cask
    extend T::Sig

    extend Forwardable
    extend Searchable
    include Metadata

    attr_reader :token, :sourcefile_path, :source, :config, :default_config

    attr_accessor :download, :allow_reassignment

    def self.all
      # TODO: uncomment for 3.7.0 and ideally avoid using ARGV by moving to e.g. CLI::Parser
      # if !ARGV.include?("--eval-all") && !Homebrew::EnvConfig.eval_all?
      #   odeprecated "Cask::Cask#all without --all or HOMEBREW_EVAL_ALL"
      # end

      Tap.flat_map(&:cask_files).map do |f|
        CaskLoader::FromTapPathLoader.new(f).load(config: nil)
      rescue CaskUnreadableError => e
        opoo e.message

        nil
      end.compact
    end

    def tap
      return super if block_given? # Object#tap

      @tap
    end

    def initialize(token, sourcefile_path: nil, source: nil, tap: nil, config: nil, allow_reassignment: false, &block)
      @token = token
      @sourcefile_path = sourcefile_path
      @source = source
      @tap = tap
      @allow_reassignment = allow_reassignment
      @block = block

      @default_config = config || Config.new

      self.config = if config_path.exist?
        Config.from_json(File.read(config_path), ignore_invalid_keys: true)
      else
        @default_config
      end
    end

    def config=(config)
      @config = config

      refresh
    end

    def refresh
      @dsl = DSL.new(self)
      return unless @block

      @dsl.instance_eval(&@block)
      @dsl.language_eval
    end

    DSL::DSL_METHODS.each do |method_name|
      define_method(method_name) { |&block| @dsl.send(method_name, &block) }
    end

    sig { returns(T::Array[[String, String]]) }
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

    def os_versions
      # TODO: use #to_hash_with_variations instead once all casks use on_system blocks
      @os_versions ||= begin
        version_os_hash = {}
        actual_version = MacOS.full_version.to_s

        MacOSVersions::SYMBOLS.each do |os_name, os_version|
          MacOS.full_version = os_version
          cask = CaskLoader.load(full_name)
          version_os_hash[os_name] = cask.version if cask.version != version
        end

        version_os_hash
      ensure
        MacOS.full_version = actual_version
      end
    end

    def full_name
      return token if tap.nil?
      return token if tap.user == "Homebrew"

      "#{tap.name}/#{token}"
    end

    def installed?
      !versions.empty?
    end

    sig { returns(T.nilable(Time)) }
    def install_time
      _, time = timestamped_versions.last
      return unless time

      Time.strptime(time, Metadata::TIMESTAMP_FORMAT)
    end

    def installed_caskfile
      installed_version = timestamped_versions.last
      metadata_main_container_path.join(*installed_version, "Casks", "#{token}.rb")
    end

    def config_path
      metadata_main_container_path/"config.json"
    end

    def checksumable?
      DownloadStrategyDetector.detect(url.to_s, url.using) <= AbstractFileDownloadStrategy
    end

    def download_sha_path
      metadata_main_container_path/"LATEST_DOWNLOAD_SHA256"
    end

    def new_download_sha
      require "cask/installer"

      # Call checksumable? before hashing
      @new_download_sha ||= Installer.new(self, verify_download_integrity: false)
                                     .download(quiet: true)
                                     .instance_eval { |x| Digest::SHA256.file(x).hexdigest }
    end

    def outdated_download_sha?
      return true unless checksumable?

      current_download_sha = download_sha_path.read if download_sha_path.exist?
      current_download_sha.blank? || current_download_sha != new_download_sha
    end

    def caskroom_path
      @caskroom_path ||= Caskroom.path.join(token)
    end

    def outdated?(greedy: false, greedy_latest: false, greedy_auto_updates: false)
      !outdated_versions(greedy: greedy, greedy_latest: greedy_latest,
                         greedy_auto_updates: greedy_auto_updates).empty?
    end

    def outdated_versions(greedy: false, greedy_latest: false, greedy_auto_updates: false)
      # special case: tap version is not available
      return [] if version.nil?

      if version.latest?
        return versions if (greedy || greedy_latest) && outdated_download_sha?

        return []
      elsif auto_updates && !greedy && !greedy_auto_updates
        return []
      end

      installed = versions
      current   = installed.last

      # not outdated unless there is a different version on tap
      return [] if current == version

      # collect all installed versions that are different than tap version and return them
      installed.reject { |v| v == version }
    end

    def outdated_info(greedy, verbose, json, greedy_latest, greedy_auto_updates)
      return token if !verbose && !json

      installed_versions = outdated_versions(greedy: greedy, greedy_latest: greedy_latest,
                                             greedy_auto_updates: greedy_auto_updates).join(", ")

      if json
        {
          name:               token,
          installed_versions: installed_versions,
          current_version:    version,
        }
      else
        "#{token} (#{installed_versions}) != #{version}"
      end
    end

    def to_s
      @token
    end

    def hash
      token.hash
    end

    def eql?(other)
      instance_of?(other.class) && token == other.token
    end
    alias == eql?

    def to_h
      {
        "token"          => token,
        "full_token"     => full_name,
        "tap"            => tap&.name,
        "name"           => name,
        "desc"           => desc,
        "homepage"       => homepage,
        "url"            => url,
        "appcast"        => appcast,
        "version"        => version,
        "versions"       => os_versions,
        "installed"      => versions.last,
        "outdated"       => outdated?,
        "sha256"         => sha256,
        "artifacts"      => artifacts_list,
        "caveats"        => (to_h_string_gsubs(caveats) unless caveats.empty?),
        "depends_on"     => depends_on,
        "conflicts_with" => conflicts_with,
        "container"      => container,
        "auto_updates"   => auto_updates,
      }
    end

    def to_hash_with_variations
      hash = to_h
      variations = {}

      hash_keys_to_skip = %w[outdated installed versions]

      if @dsl.on_system_blocks_exist?
        [:arm, :intel].each do |arch|
          MacOSVersions::SYMBOLS.each_key do |os_name|
            bottle_tag = ::Utils::Bottles::Tag.new(system: os_name, arch: arch)
            next unless bottle_tag.valid_combination?

            Homebrew::SimulateSystem.os = os_name
            Homebrew::SimulateSystem.arch = arch

            refresh

            to_h.each do |key, value|
              next if hash_keys_to_skip.include? key
              next if value.to_s == hash[key].to_s

              variations[bottle_tag.to_sym] ||= {}
              variations[bottle_tag.to_sym][key] = value
            end
          end
        end
      end

      Homebrew::SimulateSystem.clear
      refresh

      hash["variations"] = variations
      hash
    end

    private

    def artifacts_list
      artifacts.map do |artifact|
        key, value = if artifact.is_a? Artifact::AbstractFlightBlock
          artifact.summarize
        else
          [artifact.class.dsl_key, to_h_gsubs(artifact.to_args)]
        end

        { key => value }
      end
    end

    def to_h_string_gsubs(string)
      string.to_s
            .gsub(Dir.home, "$HOME")
            .gsub(HOMEBREW_PREFIX, "$(brew --prefix)")
    end

    def to_h_array_gsubs(array)
      array.to_a.map do |value|
        to_h_gsubs(value)
      end
    end

    def to_h_hash_gsubs(hash)
      hash.to_h.transform_values do |value|
        to_h_gsubs(value)
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

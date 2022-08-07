# typed: false
# frozen_string_literal: true

require "cask/cache"
require "cask/cask"
require "uri"

module Cask
  # Loads a cask from various sources.
  #
  # @api private
  module CaskLoader
    # Loads a cask from a string.
    class FromContentLoader
      attr_reader :content

      def self.can_load?(ref)
        return false unless ref.respond_to?(:to_str)

        content = ref.to_str

        token  = /(?:"[^"]*"|'[^']*')/
        curly  = /\(\s*#{token.source}\s*\)\s*\{.*\}/
        do_end = /\s+#{token.source}\s+do(?:\s*;\s*|\s+).*end/
        regex  = /\A\s*cask(?:#{curly.source}|#{do_end.source})\s*\Z/m

        content.match?(regex)
      end

      def initialize(content)
        @content = content.force_encoding("UTF-8")
      end

      def load(config:)
        @config = config

        instance_eval(content, __FILE__, __LINE__)
      end

      private

      def cask(header_token, **options, &block)
        Cask.new(header_token, source: content, **options, config: @config, &block)
      end
    end

    # Loads a cask from a path.
    class FromPathLoader < FromContentLoader
      def self.can_load?(ref)
        path = Pathname(ref)
        path.extname == ".rb" && path.expand_path.exist?
      end

      attr_reader :token, :path

      def initialize(path) # rubocop:disable Lint/MissingSuper
        path = Pathname(path).expand_path

        @token = path.basename(".rb").to_s
        @path = path
      end

      def load(config:)
        raise CaskUnavailableError.new(token, "'#{path}' does not exist.")  unless path.exist?
        raise CaskUnavailableError.new(token, "'#{path}' is not readable.") unless path.readable?
        raise CaskUnavailableError.new(token, "'#{path}' is not a file.")   unless path.file?

        @content = path.read(encoding: "UTF-8")
        @config = config

        begin
          instance_eval(content, path).tap do |cask|
            raise CaskUnreadableError.new(token, "'#{path}' does not contain a cask.") unless cask.is_a?(Cask)
          end
        rescue NameError, ArgumentError, ScriptError => e
          error = CaskUnreadableError.new(token, e.message)
          error.set_backtrace e.backtrace
          raise error
        end
      end

      private

      def cask(header_token, **options, &block)
        raise CaskTokenMismatchError.new(token, header_token) if token != header_token

        super(header_token, **options, sourcefile_path: path, &block)
      end
    end

    # Loads a cask from a URI.
    class FromURILoader < FromPathLoader
      extend T::Sig

      def self.can_load?(ref)
        uri_regex = ::URI::DEFAULT_PARSER.make_regexp
        return false unless ref.to_s.match?(Regexp.new("\\A#{uri_regex.source}\\Z", uri_regex.options))

        uri = URI(ref)
        return false unless uri
        return false unless uri.path

        true
      end

      attr_reader :url

      sig { params(url: T.any(URI::Generic, String)).void }
      def initialize(url)
        @url = URI(url)
        super Cache.path/File.basename(@url.path)
      end

      def load(config:)
        path.dirname.mkpath

        begin
          ohai "Downloading #{url}"
          curl_download url, to: path
        rescue ErrorDuringExecution
          raise CaskUnavailableError.new(token, "Failed to download #{Formatter.url(url)}.")
        end

        super
      end
    end

    # Loads a cask from a tap path.
    class FromTapPathLoader < FromPathLoader
      def self.can_load?(ref)
        super && !Tap.from_path(ref).nil?
      end

      attr_reader :tap

      def initialize(path)
        @tap = Tap.from_path(path)
        super(path)
      end

      private

      def cask(*args, &block)
        super(*args, tap: tap, &block)
      end
    end

    # Loads a cask from a specific tap.
    class FromTapLoader < FromTapPathLoader
      def self.can_load?(ref)
        ref.to_s.match?(HOMEBREW_TAP_CASK_REGEX)
      end

      def initialize(tapped_name)
        user, repo, token = tapped_name.split("/", 3)
        super Tap.fetch(user, repo).cask_dir/"#{token}.rb"
      end

      def load(config:)
        raise TapCaskUnavailableError.new(tap, token) unless tap.installed?

        super
      end
    end

    # Loads a cask from an existing {Cask} instance.
    class FromInstanceLoader
      def self.can_load?(ref)
        ref.is_a?(Cask)
      end

      def initialize(cask)
        @cask = cask
      end

      def load(config:)
        @cask
      end
    end

    # Pseudo-loader which raises an error when trying to load the corresponding cask.
    class NullLoader < FromPathLoader
      extend T::Sig

      def self.can_load?(*)
        true
      end

      sig { params(ref: T.any(String, Pathname)).void }
      def initialize(ref)
        token = File.basename(ref, ".rb")
        super CaskLoader.default_path(token)
      end

      def load(config:)
        raise CaskUnavailableError.new(token, "No Cask with this name exists.")
      end
    end

    def self.path(ref)
      self.for(ref).path
    end

    def self.load(ref, config: nil)
      self.for(ref).load(config: config)
    end

    def self.for(ref)
      [
        FromInstanceLoader,
        FromContentLoader,
        FromURILoader,
        FromTapLoader,
        FromTapPathLoader,
        FromPathLoader,
      ].each do |loader_class|
        next unless loader_class.can_load?(ref)

        if loader_class == FromTapLoader && Homebrew::EnvConfig.install_from_api? &&
           ref.start_with?("homebrew/cask/") && Homebrew::API::CaskSource.available?(ref)
          return FromContentLoader.new(Homebrew::API::CaskSource.fetch(ref))
        end

        return loader_class.new(ref)
      end

      if Homebrew::EnvConfig.install_from_api? && Homebrew::API::CaskSource.available?(ref)
        return FromContentLoader.new(Homebrew::API::CaskSource.fetch(ref))
      end

      return FromTapPathLoader.new(default_path(ref)) if FromTapPathLoader.can_load?(default_path(ref))

      case (possible_tap_casks = tap_paths(ref)).count
      when 1
        return FromTapPathLoader.new(possible_tap_casks.first)
      when 2..Float::INFINITY
        loaders = possible_tap_casks.map(&FromTapPathLoader.method(:new))

        raise TapCaskAmbiguityError.new(ref, loaders)
      end

      possible_installed_cask = Cask.new(ref)
      return FromPathLoader.new(possible_installed_cask.installed_caskfile) if possible_installed_cask.installed?

      NullLoader.new(ref)
    end

    def self.default_path(token)
      Tap.default_cask_tap.cask_dir/"#{token.to_s.downcase}.rb"
    end

    def self.tap_paths(token)
      Tap.map { |t| t.cask_dir/"#{token.to_s.downcase}.rb" }
         .select(&:exist?)
    end
  end
end

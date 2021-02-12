# typed: false
# frozen_string_literal: true

require "locale"
require "lazy_object"
require "livecheck"

require "cask/artifact"

require "cask/caskroom"
require "cask/exceptions"

require "cask/dsl/appcast"
require "cask/dsl/base"
require "cask/dsl/caveats"
require "cask/dsl/conflicts_with"
require "cask/dsl/container"
require "cask/dsl/depends_on"
require "cask/dsl/postflight"
require "cask/dsl/preflight"
require "cask/dsl/uninstall_postflight"
require "cask/dsl/uninstall_preflight"
require "cask/dsl/version"

require "cask/url"
require "cask/utils"

module Cask
  # Class representing the domain-specific language used for casks.
  #
  # @api private
  class DSL
    ORDINARY_ARTIFACT_CLASSES = [
      Artifact::Installer,
      Artifact::App,
      Artifact::Artifact,
      Artifact::AudioUnitPlugin,
      Artifact::Binary,
      Artifact::Colorpicker,
      Artifact::Dictionary,
      Artifact::Font,
      Artifact::InputMethod,
      Artifact::InternetPlugin,
      Artifact::Manpage,
      Artifact::Pkg,
      Artifact::Prefpane,
      Artifact::Qlplugin,
      Artifact::Mdimporter,
      Artifact::ScreenSaver,
      Artifact::Service,
      Artifact::StageOnly,
      Artifact::Suite,
      Artifact::VstPlugin,
      Artifact::Vst3Plugin,
      Artifact::Uninstall,
      Artifact::Zap,
    ].freeze

    ACTIVATABLE_ARTIFACT_CLASSES = (ORDINARY_ARTIFACT_CLASSES - [Artifact::StageOnly]).freeze

    ARTIFACT_BLOCK_CLASSES = [
      Artifact::PreflightBlock,
      Artifact::PostflightBlock,
    ].freeze

    DSL_METHODS = Set.new([
      :appcast,
      :artifacts,
      :auto_updates,
      :caveats,
      :conflicts_with,
      :container,
      :desc,
      :depends_on,
      :homepage,
      :language,
      :languages,
      :name,
      :sha256,
      :staged_path,
      :url,
      :version,
      :appdir,
      :discontinued?,
      :livecheck,
      :livecheckable?,
      *ORDINARY_ARTIFACT_CLASSES.map(&:dsl_key),
      *ACTIVATABLE_ARTIFACT_CLASSES.map(&:dsl_key),
      *ARTIFACT_BLOCK_CLASSES.flat_map { |klass| [klass.dsl_key, klass.uninstall_dsl_key] },
    ]).freeze

    attr_reader :cask, :token

    def initialize(cask)
      @cask = cask
      @token = cask.token
    end

    def name(*args)
      @name ||= []
      return @name if args.empty?

      @name.concat(args.flatten)
    end

    def desc(description = nil)
      set_unique_stanza(:desc, description.nil?) { description }
    end

    def set_unique_stanza(stanza, should_return)
      return instance_variable_get("@#{stanza}") if should_return

      if instance_variable_defined?("@#{stanza}")
        raise CaskInvalidError.new(cask, "'#{stanza}' stanza may only appear once.")
      end

      instance_variable_set("@#{stanza}", yield)
    rescue CaskInvalidError
      raise
    rescue => e
      raise CaskInvalidError.new(cask, "'#{stanza}' stanza failed with: #{e}")
    end

    def homepage(homepage = nil)
      set_unique_stanza(:homepage, homepage.nil?) { homepage }
    end

    def language(*args, default: false, &block)
      if args.empty?
        language_eval
      elsif block
        @language_blocks ||= {}
        @language_blocks[args] = block

        return unless default

        unless @language_blocks.default.nil?
          raise CaskInvalidError.new(cask, "Only one default language may be defined.")
        end

        @language_blocks.default = block
      else
        raise CaskInvalidError.new(cask, "No block given to language stanza.")
      end
    end

    def language_eval
      return @language_eval if defined?(@language_eval)

      return @language_eval = nil if @language_blocks.blank?

      raise CaskInvalidError.new(cask, "No default language specified.") if @language_blocks.default.nil?

      locales = cask.config.languages
                    .map do |language|
                      Locale.parse(language)
                    rescue Locale::ParserError
                      nil
                    end
                    .compact

      locales.each do |locale|
        key = locale.detect(@language_blocks.keys)

        next if key.nil?

        return @language_eval = @language_blocks[key].call
      end

      @language_eval = @language_blocks.default.call
    end

    def languages
      return [] if @language_blocks.nil?

      @language_blocks.keys.flatten
    end

    def url(*args, **options)
      caller_location = caller_locations[0]

      set_unique_stanza(:url, args.empty? && options.empty? && !block_given?) do
        if block_given?
          LazyObject.new { URL.new(*yield, from_block: true, caller_location: caller_location) }
        else
          URL.new(*args, **options, caller_location: caller_location)
        end
      end
    end

    def appcast(*args)
      set_unique_stanza(:appcast, args.empty?) { DSL::Appcast.new(*args) }
    end

    def container(*args)
      set_unique_stanza(:container, args.empty?) do
        DSL::Container.new(*args)
      end
    end

    def version(arg = nil)
      set_unique_stanza(:version, arg.nil?) do
        if !arg.is_a?(String) && arg != :latest
          raise CaskInvalidError.new(cask, "invalid 'version' value: #{arg.inspect}")
        end

        DSL::Version.new(arg)
      end
    end

    def sha256(arg = nil)
      set_unique_stanza(:sha256, arg.nil?) do
        case arg
        when :no_check
          arg
        when String
          Checksum.new(arg)
        else
          raise CaskInvalidError.new(cask, "invalid 'sha256' value: #{arg.inspect}")
        end
      end
    end

    # `depends_on` uses a load method so that multiple stanzas can be merged.
    def depends_on(*args)
      @depends_on ||= DSL::DependsOn.new
      return @depends_on if args.empty?

      begin
        @depends_on.load(*args)
      rescue RuntimeError => e
        raise CaskInvalidError.new(cask, e)
      end
      @depends_on
    end

    def conflicts_with(*args)
      # TODO: remove this constraint, and instead merge multiple conflicts_with stanzas
      set_unique_stanza(:conflicts_with, args.empty?) { DSL::ConflictsWith.new(*args) }
    end

    def artifacts
      @artifacts ||= SortedSet.new
    end

    def caskroom_path
      cask.caskroom_path
    end

    def staged_path
      return @staged_path if @staged_path

      cask_version = version || :unknown
      @staged_path = caskroom_path.join(cask_version.to_s)
    end

    def caveats(*strings, &block)
      @caveats ||= DSL::Caveats.new(cask)
      if block
        @caveats.eval_caveats(&block)
      elsif strings.any?
        strings.each do |string|
          @caveats.eval_caveats { string }
        end
      else
        return @caveats.to_s
      end
      @caveats
    end

    def discontinued?
      @caveats&.discontinued?
    end

    def auto_updates(auto_updates = nil)
      set_unique_stanza(:auto_updates, auto_updates.nil?) { auto_updates }
    end

    def livecheck(&block)
      @livecheck ||= Livecheck.new(self)
      return @livecheck unless block

      raise CaskInvalidError.new(cask, "'livecheck' stanza may only appear once.") if @livecheckable

      @livecheckable = true
      @livecheck.instance_eval(&block)
    end

    def livecheckable?
      @livecheckable == true
    end

    ORDINARY_ARTIFACT_CLASSES.each do |klass|
      define_method(klass.dsl_key) do |*args|
        if [*artifacts.map(&:class), klass].include?(Artifact::StageOnly) &&
           (artifacts.map(&:class) & ACTIVATABLE_ARTIFACT_CLASSES).any?
          raise CaskInvalidError.new(cask, "'stage_only' must be the only activatable artifact.")
        end

        artifacts.add(klass.from_args(cask, *args))
      rescue CaskInvalidError
        raise
      rescue => e
        raise CaskInvalidError.new(cask, "invalid '#{klass.dsl_key}' stanza: #{e}")
      end
    end

    ARTIFACT_BLOCK_CLASSES.each do |klass|
      [klass.dsl_key, klass.uninstall_dsl_key].each do |dsl_key|
        define_method(dsl_key) do |&block|
          artifacts.add(klass.new(cask, dsl_key => block))
        end
      end
    end

    # No need to define it as it's the default/superclass implementation.
    # rubocop:disable Style/MissingRespondToMissing
    def method_missing(method, *)
      if method
        Utils.method_missing_message(method, token)
        nil
      else
        super
      end
    end
    # rubocop:enable Style/MissingRespondToMissing

    def appdir
      cask.config.appdir
    end
  end
end

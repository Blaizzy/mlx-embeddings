# typed: false
# frozen_string_literal: true

require "digest/md5"
require "extend/cachable"
require "tab"
require "utils/bottles"

require "active_support/core_ext/hash/deep_transform_values"

# The {Formulary} is responsible for creating instances of {Formula}.
# It is not meant to be used directly from formulae.
#
# @api private
module Formulary
  extend T::Sig

  extend Cachable

  URL_START_REGEX = %r{(https?|ftp|file)://}.freeze

  sig { void }
  def self.enable_factory_cache!
    @factory_cache = true
  end

  def self.factory_cached?
    !@factory_cache.nil?
  end

  def self.formula_class_defined_from_path?(path)
    cache.key?(:path) && cache[:path].key?(path)
  end

  def self.formula_class_defined_from_api?(name)
    cache.key?(:api) && cache[:api].key?(name)
  end

  def self.formula_class_get_from_path(path)
    cache[:path].fetch(path)
  end

  def self.formula_class_get_from_api(name)
    cache[:api].fetch(name)
  end

  def self.clear_cache
    cache.each do |type, cached_objects|
      next if type == :formulary_factory

      cached_objects.each_value do |klass|
        namespace = klass.name.deconstantize
        next if namespace.deconstantize != name

        remove_const(namespace.demodulize)
      end
    end

    super
  end

  # @private
  module PathnameWriteMkpath
    refine Pathname do
      def write(content, offset = nil, **open_args)
        raise "Will not overwrite #{self}" if exist? && !offset && !open_args[:mode]&.match?(/^a\+?$/)

        dirname.mkpath

        super
      end
    end
  end

  using PathnameWriteMkpath
  def self.load_formula(name, path, contents, namespace, flags:, ignore_errors:)
    raise "Formula loading disabled by HOMEBREW_DISABLE_LOAD_FORMULA!" if Homebrew::EnvConfig.disable_load_formula?

    require "formula"
    require "ignorable"

    mod = Module.new
    remove_const(namespace) if const_defined?(namespace)
    const_set(namespace, mod)

    eval_formula = lambda do
      # Set `BUILD_FLAGS` in the formula's namespace so we can
      # access them from within the formula's class scope.
      mod.const_set(:BUILD_FLAGS, flags)
      mod.module_eval(contents, path)
    rescue NameError, ArgumentError, ScriptError, MethodDeprecatedError, MacOSVersionError => e
      if e.is_a?(Ignorable::ExceptionMixin)
        e.ignore
      else
        remove_const(namespace)
        raise FormulaUnreadableError.new(name, e)
      end
    end
    if ignore_errors
      Ignorable.hook_raise(&eval_formula)
    else
      eval_formula.call
    end

    class_name = class_s(name)

    begin
      mod.const_get(class_name)
    rescue NameError => e
      class_list = mod.constants
                      .map { |const_name| mod.const_get(const_name) }
                      .select { |const| const.is_a?(Class) }
      new_exception = FormulaClassUnavailableError.new(name, path, class_name, class_list)
      remove_const(namespace)
      raise new_exception, "", e.backtrace
    end
  end

  def self.load_formula_from_path(name, path, flags:, ignore_errors:)
    contents = path.open("r") { |f| ensure_utf8_encoding(f).read }
    namespace = "FormulaNamespace#{Digest::MD5.hexdigest(path.to_s)}"
    klass = load_formula(name, path, contents, namespace, flags: flags, ignore_errors: ignore_errors)
    cache[:path] ||= {}
    cache[:path][path] = klass
  end

  def self.load_formula_from_api(name, flags:)
    namespace = "FormulaNamespaceAPI#{Digest::MD5.hexdigest(name)}"

    mod = Module.new
    remove_const(namespace) if const_defined?(namespace)
    const_set(namespace, mod)

    mod.const_set(:BUILD_FLAGS, flags)

    class_s = Formulary.class_s(name)
    json_formula = Homebrew::API::Formula.all_formulae[name]

    if (bottle_tag = Utils::Bottles.tag.to_s.presence) &&
       (variations = json_formula["variations"].presence) &&
       (variation = variations[bottle_tag].presence)
      json_formula = json_formula.merge(variation)
    end

    uses_from_macos_names = json_formula["uses_from_macos"].map do |dep|
      next dep unless dep.is_a? Hash

      dep.keys.first
    end

    klass = Class.new(::Formula) do
      desc json_formula["desc"]
      homepage json_formula["homepage"]
      license json_formula["license"]
      revision json_formula["revision"]
      version_scheme json_formula["version_scheme"]

      if (urls_stable = json_formula["urls"]["stable"]).present?
        stable do
          url urls_stable["url"]
          version json_formula["versions"]["stable"]
        end
      end

      if (bottles_stable = json_formula["bottle"]["stable"]).present?
        bottle do
          root_url bottles_stable["root_url"]
          rebuild bottles_stable["rebuild"]
          bottles_stable["files"].each do |tag, bottle_spec|
            cellar = Formulary.convert_to_string_or_symbol bottle_spec["cellar"]
            sha256 cellar: cellar, tag.to_sym => bottle_spec["sha256"]
          end
        end
      end

      if (keg_only_reason = json_formula["keg_only_reason"]).present?
        reason = Formulary.convert_to_string_or_symbol keg_only_reason["reason"]
        keg_only reason, keg_only_reason["explanation"]
      end

      if (deprecation_date = json_formula["deprecation_date"]).present?
        reason = Formulary.convert_to_deprecate_disable_reason_string_or_symbol json_formula["deprecation_reason"]
        deprecate! date: deprecation_date, because: reason
      end

      if (disable_date = json_formula["disable_date"]).present?
        reason = Formulary.convert_to_deprecate_disable_reason_string_or_symbol json_formula["disable_reason"]
        disable! date: disable_date, because: reason
      end

      json_formula["dependencies"].each do |dep|
        next if uses_from_macos_names.include? dep

        depends_on dep
      end

      [:build, :recommended, :optional].each do |type|
        json_formula["#{type}_dependencies"].each do |dep|
          next if uses_from_macos_names.include? dep

          depends_on dep => type
        end
      end

      json_formula["uses_from_macos"].each do |dep|
        dep = dep.deep_transform_values(&:to_sym) if dep.is_a?(Hash)
        uses_from_macos dep
      end

      def install
        raise "Cannot build from source from abstract formula."
      end

      @caveats_string = json_formula["caveats"]
      def caveats
        self.class.instance_variable_get(:@caveats_string)
      end
    end

    mod.const_set(class_s, klass)

    cache[:api] ||= {}
    cache[:api][name] = klass
  end

  def self.resolve(name, spec: nil, force_bottle: false, flags: [])
    if name.include?("/") || File.exist?(name)
      f = factory(name, *spec, force_bottle: force_bottle, flags: flags)
      if f.any_version_installed?
        tab = Tab.for_formula(f)
        resolved_spec = spec || tab.spec
        f.active_spec = resolved_spec if f.send(resolved_spec)
        f.build = tab
        if f.head? && tab.tabfile
          k = Keg.new(tab.tabfile.parent)
          f.version.update_commit(k.version.version.commit) if k.version.head?
        end
      end
    else
      rack = to_rack(name)
      alias_path = factory(name, force_bottle: force_bottle, flags: flags).alias_path
      f = from_rack(rack, *spec, alias_path: alias_path, force_bottle: force_bottle, flags: flags)
    end

    # If this formula was installed with an alias that has since changed,
    # then it was specified explicitly in ARGV. (Using the alias would
    # instead have found the new formula.)
    #
    # Because of this, the user is referring to this specific formula,
    # not any formula targeted by the same alias, so in this context
    # the formula shouldn't be considered outdated if the alias used to
    # install it has changed.
    f.follow_installed_alias = false

    f
  end

  def self.ensure_utf8_encoding(io)
    io.set_encoding(Encoding::UTF_8)
  end

  def self.class_s(name)
    class_name = name.capitalize
    class_name.gsub!(/[-_.\s]([a-zA-Z0-9])/) { Regexp.last_match(1).upcase }
    class_name.tr!("+", "x")
    class_name.sub!(/(.)@(\d)/, "\\1AT\\2")
    class_name
  end

  def self.convert_to_string_or_symbol(string)
    return string[1..].to_sym if string.start_with?(":")

    string
  end

  def self.convert_to_deprecate_disable_reason_string_or_symbol(string)
    require "deprecate_disable"
    return string unless DeprecateDisable::DEPRECATE_DISABLE_REASONS.keys.map(&:to_s).include?(string)

    string.to_sym
  end

  # A {FormulaLoader} returns instances of formulae.
  # Subclasses implement loaders for particular sources of formulae.
  class FormulaLoader
    include Context

    # The formula's name
    attr_reader :name
    # The formula's ruby file's path or filename
    attr_reader :path
    # The name used to install the formula
    attr_reader :alias_path

    def initialize(name, path)
      @name = name
      @path = path.resolved_path
    end

    # Gets the formula instance.
    # `alias_path` can be overridden here in case an alias was used to refer to
    # a formula that was loaded in another way.
    def get_formula(spec, alias_path: nil, force_bottle: false, flags: [], ignore_errors: false)
      alias_path ||= self.alias_path
      klass(flags: flags, ignore_errors: ignore_errors)
        .new(name, path, spec, alias_path: alias_path, force_bottle: force_bottle)
    end

    def klass(flags:, ignore_errors:)
      load_file(flags: flags, ignore_errors: ignore_errors) unless Formulary.formula_class_defined_from_path?(path)
      Formulary.formula_class_get_from_path(path)
    end

    private

    def load_file(flags:, ignore_errors:)
      $stderr.puts "#{$PROGRAM_NAME} (#{self.class.name}): loading #{path}" if debug?
      raise FormulaUnavailableError, name unless path.file?

      Formulary.load_formula_from_path(name, path, flags: flags, ignore_errors: ignore_errors)
    end
  end

  # Loads a formula from a bottle.
  class BottleLoader < FormulaLoader
    def initialize(bottle_name)
      case bottle_name
      when URL_START_REGEX
        # The name of the formula is found between the last slash and the last hyphen.
        formula_name = File.basename(bottle_name)[/(.+)-/, 1]
        resource = Resource.new(formula_name) { url bottle_name }
        resource.specs[:bottle] = true
        downloader = resource.downloader
        cached = downloader.cached_location.exist?
        downloader.fetch
        ohai "Pouring the cached bottle" if cached
        @bottle_filename = downloader.cached_location
      else
        @bottle_filename = Pathname(bottle_name).realpath
      end
      name, full_name = Utils::Bottles.resolve_formula_names @bottle_filename
      super name, Formulary.path(full_name)
    end

    def get_formula(spec, force_bottle: false, flags: [], ignore_errors: false, **)
      formula = begin
        contents = Utils::Bottles.formula_contents @bottle_filename, name: name
        Formulary.from_contents(name, path, contents, spec, force_bottle: force_bottle,
                                flags: flags, ignore_errors: ignore_errors)
      rescue FormulaUnreadableError => e
        opoo <<~EOS
          Unreadable formula in #{@bottle_filename}:
          #{e}
        EOS
        super
      rescue BottleFormulaUnavailableError => e
        opoo <<~EOS
          #{e}
          Falling back to non-bottle formula.
        EOS
        super
      end
      formula.local_bottle_path = @bottle_filename
      formula
    end
  end

  # Loads a formula from a path to an alias.
  class AliasLoader < FormulaLoader
    def initialize(alias_path)
      path = alias_path.resolved_path
      name = path.basename(".rb").to_s
      super name, path
      @alias_path = alias_path.to_s
    end
  end

  # Loads formulae from disk using a path.
  class FromPathLoader < FormulaLoader
    def initialize(path)
      path = Pathname.new(path).expand_path
      super path.basename(".rb").to_s, path
    end
  end

  # Loads formulae from URLs.
  class FromUrlLoader < FormulaLoader
    extend T::Sig

    attr_reader :url

    sig { params(url: T.any(URI::Generic, String)).void }
    def initialize(url)
      @url = url
      uri = URI(url)
      formula = File.basename(uri.path, ".rb")
      super formula, HOMEBREW_CACHE_FORMULA/File.basename(uri.path)
    end

    def load_file(flags:, ignore_errors:)
      if %r{githubusercontent.com/[\w-]+/[\w-]+/[a-f0-9]{40}(?:/Formula)?/(?<formula_name>[\w+-.@]+).rb} =~ url
        raise UnsupportedInstallationMethod,
              "Installation of #{formula_name} from a GitHub commit URL is unsupported! " \
              "`brew extract #{formula_name}` to a stable tap on GitHub instead."
      elsif url.match?(%r{^(https?|ftp)://})
        raise UnsupportedInstallationMethod,
              "Non-checksummed download of #{name} formula file from an arbitrary URL is unsupported! " \
              "`brew extract` or `brew create` and `brew tap-new` to create a formula file in a tap " \
              "on GitHub instead."
      end
      HOMEBREW_CACHE_FORMULA.mkpath
      FileUtils.rm_f(path)
      curl_download url, to: path
      super
    rescue MethodDeprecatedError => e
      if %r{github.com/(?<user>[\w-]+)/(?<repo>[\w-]+)/} =~ url
        e.issues_url = "https://github.com/#{user}/#{repo}/issues/new"
      end
      raise
    end
  end

  # Loads tapped formulae.
  class TapLoader < FormulaLoader
    attr_reader :tap

    def initialize(tapped_name, from: nil)
      warn = [:keg, :rack].exclude?(from)
      name, path = formula_name_path(tapped_name, warn: warn)
      super name, path
    end

    def formula_name_path(tapped_name, warn: true)
      user, repo, name = tapped_name.split("/", 3).map(&:downcase)
      @tap = Tap.fetch user, repo
      formula_dir = @tap.formula_dir || @tap.path
      path = formula_dir/"#{name}.rb"

      unless path.file?
        if (possible_alias = @tap.alias_dir/name).file?
          path = possible_alias.resolved_path
          name = path.basename(".rb").to_s
        elsif (new_name = @tap.formula_renames[name]) &&
              (new_path = formula_dir/"#{new_name}.rb").file?
          old_name = name
          path = new_path
          name = new_name
          new_name = @tap.core_tap? ? name : "#{@tap}/#{name}"
        elsif (new_tap_name = @tap.tap_migrations[name])
          new_tap_user, new_tap_repo, = new_tap_name.split("/")
          new_tap_name = "#{new_tap_user}/#{new_tap_repo}"
          new_tap = Tap.fetch new_tap_name
          new_tap.install unless new_tap.installed?
          new_tapped_name = "#{new_tap_name}/#{name}"
          name, path = formula_name_path(new_tapped_name, warn: false)
          old_name = tapped_name
          new_name = new_tap.core_tap? ? name : new_tapped_name
        end

        opoo "Use #{new_name} instead of deprecated #{old_name}" if warn && old_name && new_name
      end

      [name, path]
    end

    def get_formula(spec, alias_path: nil, force_bottle: false, flags: [], ignore_errors: false)
      super
    rescue FormulaUnreadableError => e
      raise TapFormulaUnreadableError.new(tap, name, e.formula_error), "", e.backtrace
    rescue FormulaClassUnavailableError => e
      raise TapFormulaClassUnavailableError.new(tap, name, e.path, e.class_name, e.class_list), "", e.backtrace
    rescue FormulaUnavailableError => e
      raise TapFormulaUnavailableError.new(tap, name), "", e.backtrace
    end

    def load_file(flags:, ignore_errors:)
      super
    rescue MethodDeprecatedError => e
      e.issues_url = tap.issues_url || tap.to_s
      raise
    end
  end

  # Pseudo-loader which will raise a {FormulaUnavailableError} when trying to load the corresponding formula.
  class NullLoader < FormulaLoader
    def initialize(name)
      super name, Formulary.core_path(name)
    end

    def get_formula(*)
      raise FormulaUnavailableError, name
    end
  end

  # Load formulae directly from their contents.
  class FormulaContentsLoader < FormulaLoader
    # The formula's contents.
    attr_reader :contents

    def initialize(name, path, contents)
      @contents = contents
      super name, path
    end

    def klass(flags:, ignore_errors:)
      $stderr.puts "#{$PROGRAM_NAME} (#{self.class.name}): loading #{path}" if debug?
      namespace = "FormulaNamespace#{Digest::MD5.hexdigest(contents.to_s)}"
      Formulary.load_formula(name, path, contents, namespace, flags: flags, ignore_errors: ignore_errors)
    end
  end

  # Load formulae from the API.
  class FormulaAPILoader < FormulaLoader
    def initialize(name)
      super name, Formulary.core_path(name)
    end

    def klass(flags:, ignore_errors:)
      load_from_api(flags: flags) unless Formulary.formula_class_defined_from_api?(name)
      Formulary.formula_class_get_from_api(name)
    end

    private

    def load_from_api(flags:)
      $stderr.puts "#{$PROGRAM_NAME} (#{self.class.name}): loading #{name} from API" if debug?

      Formulary.load_formula_from_api(name, flags: flags)
    end
  end

  # Return a {Formula} instance for the given reference.
  # `ref` is a string containing:
  #
  # * a formula name
  # * a formula pathname
  # * a formula URL
  # * a local bottle reference
  def self.factory(
    ref, spec = :stable, alias_path: nil, from: nil,
    force_bottle: false, flags: [], ignore_errors: false
  )
    raise ArgumentError, "Formulae must have a ref!" unless ref

    cache_key = "#{ref}-#{spec}-#{alias_path}-#{from}"
    if factory_cached? && cache[:formulary_factory] &&
       cache[:formulary_factory][cache_key]
      return cache[:formulary_factory][cache_key]
    end

    formula = loader_for(ref, from: from).get_formula(spec, alias_path: alias_path,
                                                      force_bottle: force_bottle, flags: flags,
                                                      ignore_errors: ignore_errors)
    if factory_cached?
      cache[:formulary_factory] ||= {}
      cache[:formulary_factory][cache_key] ||= formula
    end
    formula
  end

  # Return a {Formula} instance for the given rack.
  #
  # @param spec when nil, will auto resolve the formula's spec.
  # @param :alias_path will be used if the formula is found not to be
  #   installed, and discarded if it is installed because the `alias_path` used
  #   to install the formula will be set instead.
  def self.from_rack(rack, spec = nil, alias_path: nil, force_bottle: false, flags: [])
    kegs = rack.directory? ? rack.subdirs.map { |d| Keg.new(d) } : []
    keg = kegs.find(&:linked?) || kegs.find(&:optlinked?) || kegs.max_by(&:version)

    if keg
      from_keg(keg, spec, alias_path: alias_path, force_bottle: force_bottle, flags: flags)
    else
      factory(rack.basename.to_s, spec || :stable, alias_path: alias_path, from: :rack,
              force_bottle: force_bottle, flags: flags)
    end
  end

  # Return whether given rack is keg-only.
  def self.keg_only?(rack)
    Formulary.from_rack(rack).keg_only?
  rescue FormulaUnavailableError, TapFormulaAmbiguityError, TapFormulaWithOldnameAmbiguityError
    false
  end

  # Return a {Formula} instance for the given keg.
  #
  # @param spec when nil, will auto resolve the formula's spec.
  def self.from_keg(keg, spec = nil, alias_path: nil, force_bottle: false, flags: [])
    tab = Tab.for_keg(keg)
    tap = tab.tap
    spec ||= tab.spec

    f = if tap.nil?
      factory(keg.rack.basename.to_s, spec, alias_path: alias_path, from: :keg,
              force_bottle: force_bottle, flags: flags)
    else
      begin
        factory("#{tap}/#{keg.rack.basename}", spec, alias_path: alias_path, from: :keg,
                force_bottle: force_bottle, flags: flags)
      rescue FormulaUnavailableError
        # formula may be migrated to different tap. Try to search in core and all taps.
        factory(keg.rack.basename.to_s, spec, alias_path: alias_path, from: :keg,
                force_bottle: force_bottle, flags: flags)
      end
    end
    f.build = tab
    f.build.used_options = Tab.remap_deprecated_options(f.deprecated_options, tab.used_options).as_flags
    f.version.update_commit(keg.version.version.commit) if f.head? && keg.version.head?
    f
  end

  # Return a {Formula} instance directly from contents.
  def self.from_contents(
    name, path, contents, spec = :stable, alias_path: nil,
    force_bottle: false, flags: [], ignore_errors: false
  )
    FormulaContentsLoader.new(name, path, contents)
                         .get_formula(spec, alias_path: alias_path, force_bottle: force_bottle,
                                      flags: flags, ignore_errors: ignore_errors)
  end

  def self.to_rack(ref)
    # If using a fully-scoped reference, check if the formula can be resolved.
    factory(ref) if ref.include? "/"

    # Check whether the rack with the given name exists.
    if (rack = HOMEBREW_CELLAR/File.basename(ref, ".rb")).directory?
      return rack.resolved_path
    end

    # Use canonical name to locate rack.
    (HOMEBREW_CELLAR/canonical_name(ref)).resolved_path
  end

  def self.canonical_name(ref)
    loader_for(ref).name
  rescue TapFormulaAmbiguityError
    # If there are multiple tap formulae with the name of ref,
    # then ref is the canonical name
    ref.downcase
  end

  def self.path(ref)
    loader_for(ref).path
  end

  def self.loader_for(ref, from: nil)
    case ref
    when HOMEBREW_BOTTLES_EXTNAME_REGEX
      return BottleLoader.new(ref)
    when URL_START_REGEX
      return FromUrlLoader.new(ref)
    when HOMEBREW_TAP_FORMULA_REGEX
      if ref.start_with?("homebrew/core/") && Homebrew::EnvConfig.install_from_api?
        name = ref.split("/", 3).last
        return FormulaAPILoader.new(name) if Homebrew::API::Formula.all_formulae.key?(name)
      end

      return TapLoader.new(ref, from: from)
    end

    return FromPathLoader.new(ref) if File.extname(ref) == ".rb" && Pathname.new(ref).expand_path.exist?

    if Homebrew::EnvConfig.install_from_api? && Homebrew::API::Formula.all_formulae.key?(ref)
      return FormulaAPILoader.new(ref)
    end

    formula_with_that_name = core_path(ref)
    return FormulaLoader.new(ref, formula_with_that_name) if formula_with_that_name.file?

    possible_alias = CoreTap.instance.alias_dir/ref
    return AliasLoader.new(possible_alias) if possible_alias.file?

    possible_tap_formulae = tap_paths(ref)
    raise TapFormulaAmbiguityError.new(ref, possible_tap_formulae) if possible_tap_formulae.size > 1

    if possible_tap_formulae.size == 1
      path = possible_tap_formulae.first.resolved_path
      name = path.basename(".rb").to_s
      return FormulaLoader.new(name, path)
    end

    return TapLoader.new("#{CoreTap.instance}/#{ref}", from: from) if CoreTap.instance.formula_renames.key?(ref)

    possible_taps = Tap.select { |tap| tap.formula_renames.key?(ref) }

    if possible_taps.size > 1
      possible_tap_newname_formulae = possible_taps.map { |tap| "#{tap}/#{tap.formula_renames[ref]}" }
      raise TapFormulaWithOldnameAmbiguityError.new(ref, possible_tap_newname_formulae)
    end

    return TapLoader.new("#{possible_taps.first}/#{ref}", from: from) unless possible_taps.empty?

    possible_keg_formula = Pathname.new("#{HOMEBREW_PREFIX}/opt/#{ref}/.brew/#{ref}.rb")
    return FormulaLoader.new(ref, possible_keg_formula) if possible_keg_formula.file?

    possible_cached_formula = Pathname.new("#{HOMEBREW_CACHE_FORMULA}/#{ref}.rb")
    return FormulaLoader.new(ref, possible_cached_formula) if possible_cached_formula.file?

    NullLoader.new(ref)
  end

  def self.core_path(name)
    CoreTap.instance.formula_dir/"#{name.to_s.downcase}.rb"
  end

  def self.tap_paths(name, taps = Dir[HOMEBREW_LIBRARY/"Taps/*/*/"])
    name = name.to_s.downcase
    taps.map do |tap|
      Pathname.glob([
        "#{tap}Formula/#{name}.rb",
        "#{tap}HomebrewFormula/#{name}.rb",
        "#{tap}#{name}.rb",
        "#{tap}Aliases/#{name}",
      ]).find(&:file?)
    end.compact
  end
end

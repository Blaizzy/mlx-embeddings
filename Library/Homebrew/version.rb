# typed: true
# frozen_string_literal: true

require "pkg_version"
require "version/null"
require "version/parser"

# A formula's version.
#
# @api private
class Version
  extend T::Sig

  include Comparable

  sig { params(name: T.any(String, Symbol), full: T::Boolean).returns(Regexp) }
  def self.formula_optionally_versioned_regex(name, full: true)
    /#{"^" if full}#{Regexp.escape(name)}(@\d[\d.]*)?#{"$" if full}/
  end

  # A part of a {Version}.
  class Token
    extend T::Sig
    extend T::Helpers
    abstract!

    include Comparable

    sig { params(val: String).returns(Token) }
    def self.create(val)
      raise TypeError, "Token value must be a string; got a #{val.class} (#{val})" unless val.respond_to?(:to_str)

      case val
      when /\A#{AlphaToken::PATTERN}\z/o   then AlphaToken
      when /\A#{BetaToken::PATTERN}\z/o    then BetaToken
      when /\A#{RCToken::PATTERN}\z/o      then RCToken
      when /\A#{PreToken::PATTERN}\z/o     then PreToken
      when /\A#{PatchToken::PATTERN}\z/o   then PatchToken
      when /\A#{PostToken::PATTERN}\z/o    then PostToken
      when /\A#{NumericToken::PATTERN}\z/o then NumericToken
      when /\A#{StringToken::PATTERN}\z/o  then StringToken
      else raise "Cannot find a matching token pattern"
      end.new(val)
    end

    sig { params(val: T.untyped).returns(T.nilable(Token)) }
    def self.from(val)
      return NULL_TOKEN if val.nil? || (val.respond_to?(:null?) && val.null?)

      case val
      when Token   then val
      when String  then Token.create(val)
      when Integer then Token.create(val.to_s)
      end
    end

    sig { returns(T.nilable(T.any(String, Integer))) }
    attr_reader :value

    sig { params(value: T.nilable(T.any(String, Integer))).void }
    def initialize(value)
      @value = T.let(value, T.untyped)
    end

    sig { abstract.params(other: T.untyped).returns(T.nilable(Integer)) }
    def <=>(other); end

    sig { returns(String) }
    def inspect
      "#<#{self.class.name} #{value.inspect}>"
    end

    sig { returns(Integer) }
    def hash
      value.hash
    end

    sig { returns(Float) }
    def to_f
      value.to_f
    end

    sig { returns(Integer) }
    def to_i
      value.to_i
    end

    sig { returns(String) }
    def to_s
      value.to_s
    end
    alias to_str to_s

    sig { returns(T::Boolean) }
    def numeric?
      false
    end

    sig { returns(T::Boolean) }
    def null?
      false
    end
  end

  # A pseudo-token representing the absence of a token.
  class NullToken < Token
    extend T::Sig

    sig { override.returns(NilClass) }
    attr_reader :value

    sig { void }
    def initialize
      super(nil)
    end

    sig { override.params(other: T.untyped).returns(T.nilable(Integer)) }
    def <=>(other)
      return unless (other = Token.from(other))

      case other
      when NullToken
        0
      when NumericToken
        other.value.zero? ? 0 : -1
      when AlphaToken, BetaToken, PreToken, RCToken
        1
      else
        -1
      end
    end

    sig { override.returns(T::Boolean) }
    def null?
      true
    end

    sig { override.returns(String) }
    def inspect
      "#<#{self.class.name}>"
    end
  end
  private_constant :NullToken

  # Represents the absence of a token.
  NULL_TOKEN = NullToken.new.freeze

  # A token string.
  class StringToken < Token
    PATTERN = /[a-z]+/i.freeze

    sig { override.returns(String) }
    attr_reader :value

    sig { params(value: String).void }
    def initialize(value)
      super(value.to_s)
    end

    sig { override.params(other: T.untyped).returns(T.nilable(Integer)) }
    def <=>(other)
      return unless (other = Token.from(other))

      case other
      when StringToken
        value <=> other.value
      when NumericToken, NullToken
        -T.must(other <=> self)
      end
    end
  end

  # A token consisting of only numbers.
  class NumericToken < Token
    PATTERN = /[0-9]+/i.freeze
    extend T::Sig

    sig { override.returns(Integer) }
    attr_reader :value

    sig { params(value: T.any(String, Integer)).void }
    def initialize(value)
      super(value.to_i)
    end

    sig { override.params(other: T.untyped).returns(T.nilable(Integer)) }
    def <=>(other)
      return unless (other = Token.from(other))

      case other
      when NumericToken
        value <=> other.value
      when StringToken
        1
      when NullToken
        -T.must(other <=> self)
      end
    end

    sig { override.returns(T::Boolean) }
    def numeric?
      true
    end
  end

  # A token consisting of an alphabetic and a numeric part.
  class CompositeToken < StringToken
    sig { returns(Integer) }
    def rev
      value[/[0-9]+/].to_i
    end
  end

  # A token representing the part of a version designating it as an alpha release.
  class AlphaToken < CompositeToken
    PATTERN = /alpha[0-9]*|a[0-9]+/i.freeze

    sig { override.params(other: T.untyped).returns(T.nilable(Integer)) }
    def <=>(other)
      return unless (other = Token.from(other))

      case other
      when AlphaToken
        rev <=> other.rev
      when BetaToken, RCToken, PreToken, PatchToken, PostToken
        -1
      else
        super
      end
    end
  end

  # A token representing the part of a version designating it as a beta release.
  class BetaToken < CompositeToken
    PATTERN = /beta[0-9]*|b[0-9]+/i.freeze

    sig { override.params(other: T.untyped).returns(T.nilable(Integer)) }
    def <=>(other)
      return unless (other = Token.from(other))

      case other
      when BetaToken
        rev <=> other.rev
      when AlphaToken
        1
      when PreToken, RCToken, PatchToken, PostToken
        -1
      else
        super
      end
    end
  end

  # A token representing the part of a version designating it as a pre-release.
  class PreToken < CompositeToken
    PATTERN = /pre[0-9]*/i.freeze

    sig { override.params(other: T.untyped).returns(T.nilable(Integer)) }
    def <=>(other)
      return unless (other = Token.from(other))

      case other
      when PreToken
        rev <=> other.rev
      when AlphaToken, BetaToken
        1
      when RCToken, PatchToken, PostToken
        -1
      else
        super
      end
    end
  end

  # A token representing the part of a version designating it as a release candidate.
  class RCToken < CompositeToken
    PATTERN = /rc[0-9]*/i.freeze

    sig { override.params(other: T.untyped).returns(T.nilable(Integer)) }
    def <=>(other)
      return unless (other = Token.from(other))

      case other
      when RCToken
        rev <=> other.rev
      when AlphaToken, BetaToken, PreToken
        1
      when PatchToken, PostToken
        -1
      else
        super
      end
    end
  end

  # A token representing the part of a version designating it as a patch release.
  class PatchToken < CompositeToken
    PATTERN = /p[0-9]*/i.freeze

    sig { override.params(other: T.untyped).returns(T.nilable(Integer)) }
    def <=>(other)
      return unless (other = Token.from(other))

      case other
      when PatchToken
        rev <=> other.rev
      when AlphaToken, BetaToken, RCToken, PreToken
        1
      else
        super
      end
    end
  end

  # A token representing the part of a version designating it as a post release.
  class PostToken < CompositeToken
    PATTERN = /.post[0-9]+/i.freeze

    sig { override.params(other: T.untyped).returns(T.nilable(Integer)) }
    def <=>(other)
      return unless (other = Token.from(other))

      case other
      when PostToken
        rev <=> other.rev
      when AlphaToken, BetaToken, RCToken, PreToken
        1
      else
        super
      end
    end
  end

  SCAN_PATTERN = Regexp.union(
    AlphaToken::PATTERN,
    BetaToken::PATTERN,
    PreToken::PATTERN,
    RCToken::PATTERN,
    PatchToken::PATTERN,
    PostToken::PATTERN,
    NumericToken::PATTERN,
    StringToken::PATTERN,
  ).freeze
  private_constant :SCAN_PATTERN

  sig { params(url: T.any(String, Pathname), specs: T.untyped).returns(Version) }
  def self.detect(url, **specs)
    parse(specs.fetch(:tag, url), detected_from_url: true)
  end

  sig { params(val: String).returns(Version) }
  def self.create(val)
    raise TypeError, "Version value must be a string; got a #{val.class} (#{val})" unless val.respond_to?(:to_str)

    if val.to_str.start_with?("HEAD")
      HeadVersion.new(val)
    else
      Version.new(val)
    end
  end

  sig { params(spec: T.any(String, Pathname), detected_from_url: T::Boolean).returns(Version) }
  def self.parse(spec, detected_from_url: false)
    version = _parse(spec, detected_from_url: detected_from_url)
    version.nil? ? NULL : new(version, detected_from_url: detected_from_url)
  end

  sig { params(spec: T.any(String, Pathname), detected_from_url: T::Boolean).returns(T.nilable(String)) }
  def self._parse(spec, detected_from_url:)
    spec = CGI.unescape(spec.to_s) if detected_from_url

    spec = Pathname.new(spec) unless spec.is_a? Pathname

    VERSION_PARSERS.each do |parser|
      version = parser.parse(spec)
      return version if version.present?
    end

    nil
  end
  private_class_method :_parse

  NUMERIC_WITH_OPTIONAL_DOTS = /(?:\d+(?:\.\d+)*)/.source.freeze
  private_constant :NUMERIC_WITH_OPTIONAL_DOTS

  NUMERIC_WITH_DOTS = /(?:\d+(?:\.\d+)+)/.source.freeze
  private_constant :NUMERIC_WITH_DOTS

  MINOR_OR_PATCH = /(?:\d+(?:\.\d+){1,2})/.source.freeze
  private_constant :MINOR_OR_PATCH

  CONTENT_SUFFIX = /(?:[._-](?i:bin|dist|stable|src|sources?|final|full))/.source.freeze
  private_constant :CONTENT_SUFFIX

  PRERELEASE_SUFFIX = /(?:[._-]?(?i:alpha|beta|pre|rc)\.?\d{,2})/.source.freeze
  private_constant :PRERELEASE_SUFFIX

  VERSION_PARSERS = [
    # date-based versioning
    # e.g. ltopers-v2017-04-14.tar.gz
    StemParser.new(/-v?(\d{4}-\d{2}-\d{2})/),

    # GitHub tarballs
    # e.g. https://github.com/foo/bar/tarball/v1.2.3
    # e.g. https://github.com/sam-github/libnet/tarball/libnet-1.1.4
    # e.g. https://github.com/isaacs/npm/tarball/v0.2.5-1
    # e.g. https://github.com/petdance/ack/tarball/1.93_02
    UrlParser.new(%r{github\.com/.+/(?:zip|tar)ball/(?:v|\w+-)?((?:\d+[._-])+\d*)$}),

    # e.g. https://github.com/erlang/otp/tarball/OTP_R15B01 (erlang style)
    UrlParser.new(/[_-]([Rr]\d+[AaBb]\d*(?:-\d+)?)/),

    # e.g. boost_1_39_0
    StemParser.new(/((?:\d+_)+\d+)$/) { |s| s.tr("_", ".") },

    # e.g. foobar-4.5.1-1
    # e.g. unrtf_0.20.4-1
    # e.g. ruby-1.9.1-p243
    StemParser.new(/[_-](#{NUMERIC_WITH_DOTS}-(?:p|P|rc|RC)?\d+)#{CONTENT_SUFFIX}?$/),

    # Hyphenated versions without software-name prefix (e.g. brew-)
    # e.g. v0.0.8-12.tar.gz
    # e.g. 3.3.04-1.tar.gz
    # e.g. v2.1-20210510.tar.gz
    # e.g. 2020.11.11-3.tar.gz
    # e.g. v3.6.6-0.2
    StemParser.new(/^v?(#{NUMERIC_WITH_DOTS}(?:-#{NUMERIC_WITH_OPTIONAL_DOTS})+)/),

    # URL with no extension
    # e.g. https://waf.io/waf-1.8.12
    # e.g. https://codeload.github.com/gsamokovarov/jump/tar.gz/v0.7.1
    UrlParser.new(/[-v](#{NUMERIC_WITH_OPTIONAL_DOTS})$/),

    # e.g. lame-398-1
    StemParser.new(/-(\d+-\d+)/),

    # e.g. foobar-4.5.1
    StemParser.new(/-(#{NUMERIC_WITH_OPTIONAL_DOTS})$/),

    # e.g. foobar-4.5.1.post1
    StemParser.new(/-(#{NUMERIC_WITH_OPTIONAL_DOTS}(.post\d+)?)$/),

    # e.g. foobar-4.5.1b
    StemParser.new(/-(#{NUMERIC_WITH_OPTIONAL_DOTS}(?:[abc]|rc|RC)\d*)$/),

    # e.g. foobar-4.5.0-alpha5, foobar-4.5.0-beta1, or foobar-4.50-beta
    StemParser.new(/-(#{NUMERIC_WITH_OPTIONAL_DOTS}-(?:alpha|beta|rc)\d*)$/),

    # e.g. https://ftpmirror.gnu.org/libidn/libidn-1.29-win64.zip
    # e.g. https://ftpmirror.gnu.org/libmicrohttpd/libmicrohttpd-0.9.17-w32.zip
    StemParser.new(/-(#{MINOR_OR_PATCH})-w(?:in)?(?:32|64)$/),

    # Opam packages
    # e.g. https://opam.ocaml.org/archives/sha.1.9+opam.tar.gz
    # e.g. https://opam.ocaml.org/archives/lablgtk.2.18.3+opam.tar.gz
    # e.g. https://opam.ocaml.org/archives/easy-format.1.0.2+opam.tar.gz
    StemParser.new(/\.(#{MINOR_OR_PATCH})\+opam$/),

    # e.g. https://ftpmirror.gnu.org/mtools/mtools-4.0.18-1.i686.rpm
    # e.g. https://ftpmirror.gnu.org/autogen/autogen-5.5.7-5.i386.rpm
    # e.g. https://ftpmirror.gnu.org/libtasn1/libtasn1-2.8-x86.zip
    # e.g. https://ftpmirror.gnu.org/libtasn1/libtasn1-2.8-x64.zip
    # e.g. https://ftpmirror.gnu.org/mtools/mtools_4.0.18_i386.deb
    StemParser.new(/[_-](#{MINOR_OR_PATCH}(?:-\d+)?)[._-](?:i[36]86|x86|x64(?:[_-](?:32|64))?)$/),

    # e.g. https://registry.npmjs.org/@angular/cli/-/cli-1.3.0-beta.1.tgz
    # e.g. https://github.com/dlang/dmd/archive/v2.074.0-beta1.tar.gz
    # e.g. https://github.com/dlang/dmd/archive/v2.074.0-rc1.tar.gz
    # e.g. https://github.com/premake/premake-core/releases/download/v5.0.0-alpha10/premake-5.0.0-alpha10-src.zip
    StemParser.new(/[.-vV]?(#{NUMERIC_WITH_DOTS}#{PRERELEASE_SUFFIX})/),

    # e.g. foobar4.5.1
    StemParser.new(/(#{NUMERIC_WITH_OPTIONAL_DOTS})$/),

    # e.g. foobar-4.5.0-bin
    StemParser.new(/[-vV](#{NUMERIC_WITH_DOTS}[abc]?)#{CONTENT_SUFFIX}$/),

    # dash version style
    # e.g. http://www.antlr.org/download/antlr-3.4-complete.jar
    # e.g. https://cdn.nuxeo.com/nuxeo-9.2/nuxeo-server-9.2-tomcat.zip
    # e.g. https://search.maven.org/remotecontent?filepath=com/facebook/presto/presto-cli/0.181/presto-cli-0.181-executable.jar
    # e.g. https://search.maven.org/remotecontent?filepath=org/fusesource/fuse-extra/fusemq-apollo-mqtt/1.3/fusemq-apollo-mqtt-1.3-uber.jar
    # e.g. https://search.maven.org/remotecontent?filepath=org/apache/orc/orc-tools/1.2.3/orc-tools-1.2.3-uber.jar
    StemParser.new(/-(#{NUMERIC_WITH_DOTS})-/),

    # e.g. dash_0.5.5.1.orig.tar.gz (Debian style)
    StemParser.new(/_(#{NUMERIC_WITH_DOTS}[abc]?)\.orig$/),

    # e.g. https://www.openssl.org/source/openssl-0.9.8s.tar.gz
    StemParser.new(/-v?(\d[^-]+)/),

    # e.g. astyle_1.23_macosx.tar.gz
    StemParser.new(/_v?(\d[^_]+)/),

    # e.g. http://mirrors.jenkins-ci.org/war/1.486/jenkins.war
    # e.g. https://github.com/foo/bar/releases/download/0.10.11/bar.phar
    # e.g. https://github.com/clojure/clojurescript/releases/download/r1.9.293/cljs.jar
    # e.g. https://github.com/fibjs/fibjs/releases/download/v0.6.1/fullsrc.zip
    # e.g. https://wwwlehre.dhbw-stuttgart.de/~sschulz/WORK/E_DOWNLOAD/V_1.9/E.tgz
    # e.g. https://github.com/JustArchi/ArchiSteamFarm/releases/download/2.3.2.0/ASF.zip
    # e.g. https://people.gnome.org/~newren/eg/download/1.7.5.2/eg
    UrlParser.new(%r{/(?:[rvV]_?)?(\d+\.\d+(?:\.\d+){,2})}),

    # e.g. https://www.ijg.org/files/jpegsrc.v8d.tar.gz
    StemParser.new(/\.v(\d+[a-z]?)/),

    # e.g. https://secure.php.net/get/php-7.1.10.tar.bz2/from/this/mirror
    UrlParser.new(/[.-vV]?(#{NUMERIC_WITH_DOTS}#{PRERELEASE_SUFFIX}?)/),
  ].freeze
  private_constant :VERSION_PARSERS

  sig { params(val: T.any(PkgVersion, String, Version), detected_from_url: T::Boolean).void }
  def initialize(val, detected_from_url: false)
    raise TypeError, "Version value must be a string; got a #{val.class} (#{val})" unless val.respond_to?(:to_str)

    @version = val.to_str
    @detected_from_url = detected_from_url
  end

  sig { returns(T::Boolean) }
  def detected_from_url?
    @detected_from_url
  end

  sig { returns(T::Boolean) }
  def head?
    false
  end

  sig { returns(T::Boolean) }
  def null?
    false
  end

  sig { params(other: T.untyped).returns(T.nilable(Integer)) }
  def <=>(other)
    # Needed to retain API compatibility with older string comparisons
    # for compiler versions, etc.
    other = Version.new(other) if other.is_a? String
    # Used by the *_build_version comparisons, which formerly returned Fixnum
    other = Version.new(other.to_s) if other.is_a? Integer
    return 1 if other.nil?
    return 1 if other.respond_to?(:null?) && other.null?

    other = Version.new(other.to_s) if other.is_a? Token
    return unless other.is_a?(Version)
    return 0 if version == other.version
    return 1 if head? && !other.head?
    return -1 if !head? && other.head?
    return 0 if head? && other.head?

    ltokens = tokens
    rtokens = other.tokens
    max = max(ltokens.length, rtokens.length)
    l = r = 0

    while l < max
      a = ltokens[l] || NULL_TOKEN
      b = rtokens[r] || NULL_TOKEN

      if a == b
        l += 1
        r += 1
        next
      elsif a.numeric? && !b.numeric?
        return 1 if a > NULL_TOKEN

        l += 1
      elsif !a.numeric? && b.numeric?
        return -1 if b > NULL_TOKEN

        r += 1
      else
        return a <=> b
      end
    end

    0
  end
  alias eql? ==

  # @api public
  sig { returns(T.nilable(Token)) }
  def major
    tokens.first
  end

  # @api public
  sig { returns(T.nilable(Token)) }
  def minor
    tokens.second
  end

  # @api public
  sig { returns(T.nilable(Token)) }
  def patch
    tokens.third
  end

  # @api public
  sig { returns(T.self_type) }
  def major_minor
    self.class.new([major, minor].compact.join("."))
  end

  # @api public
  sig { returns(T.self_type) }
  def major_minor_patch
    self.class.new([major, minor, patch].compact.join("."))
  end

  sig { returns(T::Boolean) }
  def empty?
    version.empty?
  end

  sig { returns(Integer) }
  def hash
    version.hash
  end

  sig { returns(Float) }
  def to_f
    version.to_f
  end

  sig { returns(Integer) }
  def to_i
    version.to_i
  end

  sig { returns(String) }
  def to_s
    version.dup
  end
  alias to_str to_s

  protected

  sig { returns(String) }
  attr_reader :version

  sig { returns(T::Array[Token]) }
  def tokens
    @tokens ||= tokenize
  end

  private

  sig { params(a: Integer, b: Integer).returns(Integer) }
  def max(a, b)
    (a > b) ? a : b
  end

  sig { returns(T::Array[Token]) }
  def tokenize
    version.scan(SCAN_PATTERN).map { |token| Token.create(T.cast(token, String)) }
  end
end

# A formula's HEAD version.
# @see https://docs.brew.sh/Formula-Cookbook#unstable-versions-head Unstable versions (head)
#
# @api private
class HeadVersion < Version
  extend T::Sig

  sig { returns(T.nilable(String)) }
  attr_reader :commit

  def initialize(*)
    super
    @commit = @version[/^HEAD-(.+)$/, 1]
  end

  sig { params(commit: T.nilable(String)).void }
  def update_commit(commit)
    @commit = commit
    @version = if commit
      "HEAD-#{commit}"
    else
      "HEAD"
    end
  end

  sig { returns(T::Boolean) }
  def head?
    true
  end
end

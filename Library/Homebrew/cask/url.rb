# typed: true
# frozen_string_literal: true

# Class corresponding to the `url` stanza.
#
# @api private
class URL < Delegator
  extend T::Sig

  # @api private
  class DSL
    extend T::Sig

    attr_reader :uri, :specs,
                :verified, :using,
                :tag, :branch, :revisions, :revision,
                :trust_cert, :cookies, :referer, :header, :user_agent,
                :data

    extend Forwardable
    def_delegators :uri, :path, :scheme, :to_s

    # @api public
    sig {
      params(
        uri:        T.any(URI::Generic, String),
        verified:   T.nilable(String),
        using:      T.nilable(Symbol),
        tag:        T.nilable(String),
        branch:     T.nilable(String),
        revisions:  T.nilable(T::Array[String]),
        revision:   T.nilable(String),
        trust_cert: T.nilable(T::Boolean),
        cookies:    T.nilable(T::Hash[String, String]),
        referer:    T.nilable(T.any(URI::Generic, String)),
        header:     T.nilable(String),
        user_agent: T.nilable(T.any(Symbol, String)),
        data:       T.nilable(T::Hash[String, String]),
      ).void
    }
    def initialize(
      uri,
      verified: nil,
      using: nil,
      tag: nil,
      branch: nil,
      revisions: nil,
      revision: nil,
      trust_cert: nil,
      cookies: nil,
      referer: nil,
      header: nil,
      user_agent: nil,
      data: nil
    )

      @uri = URI(uri)

      specs = {}
      specs[:verified]   = @verified   = verified
      specs[:using]      = @using      = using
      specs[:tag]        = @tag        = tag
      specs[:branch]     = @branch     = branch
      specs[:revisions]  = @revisions  = revisions
      specs[:revision]   = @revision   = revision
      specs[:trust_cert] = @trust_cert = trust_cert
      specs[:cookies]    = @cookies    = cookies
      specs[:referer]    = @referer    = referer
      specs[:header]     = @header     = header
      specs[:user_agent] = @user_agent = user_agent || :default
      specs[:data]       = @data       = data

      @specs = specs.compact
    end
  end

  # @api private
  class BlockDSL
    extend T::Sig

    module PageWithURL
      extend T::Sig

      # @api public
      sig { returns(URI::Generic) }
      attr_accessor :url
    end

    sig {
      params(
        uri:   T.nilable(T.any(URI::Generic, String)),
        dsl:   T.nilable(Cask::DSL),
        block: T.proc.params(arg0: T.all(String, PageWithURL)).returns(T.untyped),
      ).void
    }
    def initialize(uri, dsl: nil, &block)
      @uri = uri
      @dsl = dsl
      @block = block
    end

    sig { returns(T.untyped) }
    def call
      if @uri
        result = curl_output("--fail", "--silent", "--location", @uri)
        result.assert_success!

        page = result.stdout
        page.extend PageWithURL
        page.url = URI(@uri)

        instance_exec(page, &@block)
      else
        instance_exec(&@block)
      end
    end

    # @api public
    sig {
      params(
        uri:   T.any(URI::Generic, String),
        block: T.proc.params(arg0: T.all(String, PageWithURL)).returns(T.untyped),
      ).void
    }
    def url(uri, &block)
      self.class.new(uri, &block).call
    end
    private :url

    # @api public
    def method_missing(method, *args, &block)
      if @dsl.respond_to?(method)
        T.unsafe(@dsl).public_send(method, *args, &block)
      else
        super
      end
    end

    def respond_to_missing?(method, include_all)
      @dsl.respond_to?(method, include_all) || super
    end
  end

  sig {
    params(
      uri:             T.nilable(T.any(URI::Generic, String)),
      verified:        T.nilable(String),
      using:           T.nilable(Symbol),
      tag:             T.nilable(String),
      branch:          T.nilable(String),
      revisions:       T.nilable(T::Array[String]),
      revision:        T.nilable(String),
      trust_cert:      T.nilable(T::Boolean),
      cookies:         T.nilable(T::Hash[String, String]),
      referer:         T.nilable(T.any(URI::Generic, String)),
      header:          T.nilable(String),
      user_agent:      T.nilable(T.any(Symbol, String)),
      data:            T.nilable(T::Hash[String, String]),
      caller_location: Thread::Backtrace::Location,
      dsl:             T.nilable(Cask::DSL),
      block:           T.nilable(T.proc.params(arg0: T.all(String, BlockDSL::PageWithURL)).returns(T.untyped)),
    ).void
  }
  def initialize(
    uri = nil,
    verified: nil,
    using: nil,
    tag: nil,
    branch: nil,
    revisions: nil,
    revision: nil,
    trust_cert: nil,
    cookies: nil,
    referer: nil,
    header: nil,
    user_agent: nil,
    data: nil,
    caller_location: T.must(caller_locations).fetch(0),
    dsl: nil,
    &block
  )
    super(
    if block
      LazyObject.new do
        *args = BlockDSL.new(uri, dsl: dsl, &block).call
        options = args.last.is_a?(Hash) ? args.pop : {}
        uri = T.let(args.first, T.any(URI::Generic, String))
        DSL.new(uri, **options)
      end
    else
      DSL.new(
        T.must(uri),
        verified:   verified,
        using:      using,
        tag:        tag,
        branch:     branch,
        revisions:  revisions,
        revision:   revision,
        trust_cert: trust_cert,
        cookies:    cookies,
        referer:    referer,
        header:     header,
        user_agent: user_agent,
        data:       data,
      )
    end
    )

    @from_block = !block.nil?
    @caller_location = caller_location
  end

  def __getobj__
    @dsl
  end

  def __setobj__(dsl)
    @dsl = dsl
  end

  sig { returns(T.nilable(String)) }
  def raw_interpolated_url
    return @raw_interpolated_url if defined?(@raw_interpolated_url)

    @raw_interpolated_url =
      Pathname(@caller_location.absolute_path)
      .each_line.drop(@caller_location.lineno - 1)
      .first&.yield_self { |line| line[/url\s+"([^"]+)"/, 1] }
  end
  private :raw_interpolated_url

  sig { params(ignore_major_version: T::Boolean).returns(T::Boolean) }
  def unversioned?(ignore_major_version: false)
    interpolated_url = raw_interpolated_url

    return false unless interpolated_url

    interpolated_url = interpolated_url.gsub(/\#{\s*version\s*\.major\s*}/, "") if ignore_major_version

    interpolated_url.exclude?('#{')
  end

  sig { returns(T::Boolean) }
  def from_block?
    @from_block
  end
end

# typed: true
# frozen_string_literal: true

# Class corresponding to the `url` stanza.
#
# @api private
class URL
  extend T::Sig

  attr_reader :uri, :specs,
              :verified, :using,
              :tag, :branch, :revisions, :revision,
              :trust_cert, :cookies, :referer, :header, :user_agent,
              :data

  extend Forwardable
  def_delegators :uri, :path, :scheme, :to_s

  sig {
    params(
      uri:             T.any(URI::Generic, String),
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
      from_block:      T::Boolean,
      caller_location: Thread::Backtrace::Location,
    ).returns(T.untyped)
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
    data: nil,
    from_block: false,
    caller_location: T.must(caller_locations).fetch(0)
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

    @from_block = from_block
    @caller_location = caller_location
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

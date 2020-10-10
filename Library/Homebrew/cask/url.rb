# typed: strict
# frozen_string_literal: true

# Class corresponding to the `url` stanza.
#
# @api private
class URL
  extend T::Sig

  attr_reader :uri, :specs,
              :using,
              :tag, :branch, :revisions, :revision,
              :trust_cert, :cookies, :referer, :user_agent,
              :data

  extend Forwardable
  def_delegators :uri, :path, :scheme, :to_s

  sig do
    params(
      uri:        T.any(URI::Generic, String),
      using:      T.nilable(Symbol),
      tag:        T.nilable(String),
      branch:     T.nilable(String),
      revisions:  T.nilable(T::Array[String]),
      revision:   T.nilable(String),
      trust_cert: T.nilable(T::Boolean),
      cookies:    T.nilable(T::Hash[String, String]),
      referer:    T.nilable(T.any(URI::Generic, String)),
      user_agent: T.nilable(T.any(Symbol, String)),
      data:       T.nilable(T::Hash[String, String]),
    ).returns(T.untyped)
  end
  def initialize(
    uri,
    using: nil,
    tag: nil,
    branch: nil,
    revisions: nil,
    revision: nil,
    trust_cert: nil,
    cookies: nil,
    referer: nil,
    user_agent: nil,
    data: nil
  )
    @uri = URI(uri)

    specs = {}
    specs[:using]      = @using      = using
    specs[:tag]        = @tag        = tag
    specs[:branch]     = @branch     = branch
    specs[:revisions]  = @revisions  = revisions
    specs[:revision]   = @revision   = revision
    specs[:trust_cert] = @trust_cert = trust_cert
    specs[:cookies]    = @cookies    = cookies
    specs[:referer]    = @referer    = referer
    specs[:user_agent] = @user_agent = user_agent || :default
    specs[:data]       = @data       = data

    @specs = specs.compact
  end
end

# typed: true
# frozen_string_literal: true

class Version
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

end

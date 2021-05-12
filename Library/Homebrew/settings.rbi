# typed: strict

module Homebrew
  module Settings
    include Kernel

    sig { params(setting: T.any(String, Symbol), repo: Pathname).returns(T.nilable(String)) }
    def read(setting, repo: HOMEBREW_REPOSITORY); end

    sig { params(setting: T.any(String, Symbol), value: T.any(String, T::Boolean), repo: Pathname).void }
    def write(setting, value, repo: HOMEBREW_REPOSITORY); end

    sig { params(setting: T.any(String, Symbol), repo: Pathname).void }
    def delete(setting, repo: HOMEBREW_REPOSITORY); end
  end
end

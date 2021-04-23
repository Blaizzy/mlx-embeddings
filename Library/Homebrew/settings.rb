# typed: true
# frozen_string_literal: true

require "system_command"

module Homebrew
  # Helper functions for reading and writing settings.
  #
  # @api private
  module Settings
    extend T::Sig
    include SystemCommand::Mixin

    module_function

    sig { params(setting: T.any(String, Symbol), repo: Pathname).returns(T.nilable(String)) }
    def read(setting, repo: HOMEBREW_REPOSITORY)
      return unless (repo/".git/config").exist?

      system_command("git", args: ["config", "--get", "homebrew.#{setting}"], chdir: repo).stdout.chomp.presence
    end

    sig { params(setting: T.any(String, Symbol), value: T.any(String, T::Boolean), repo: Pathname).void }
    def write(setting, value, repo: HOMEBREW_REPOSITORY)
      return unless (repo/".git/config").exist?

      value = value.to_s

      return if read(setting, repo: repo) == value

      system_command! "git", args: ["config", "--replace-all", "homebrew.#{setting}", value], chdir: repo
    end

    sig { params(setting: T.any(String, Symbol), repo: Pathname).void }
    def delete(setting, repo: HOMEBREW_REPOSITORY)
      return unless (repo/".git/config").exist?

      return if read(setting, repo: repo).blank?

      system_command! "git", args: ["config", "--unset-all", "homebrew.#{setting}"], chdir: repo
    end
  end
end

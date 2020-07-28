# frozen_string_literal: true

require "cli/parser"

module Cask
  class Cmd
    class Audit < AbstractCommand
      option "--download",        :download_arg,        false
      option "--appcast",         :appcast_arg,         false
      option "--token-conflicts", :token_conflicts_arg, false
      option "--strict",          :strict_arg,          false
      option "--online",          :online_arg,          false
      option "--new-cask",        :new_cask_arg,        false

      def self.usage
        <<~EOS
          `cask audit` [<options>] [<cask>]

          --strict          - Run additional, stricter style checks.
          --online          - Run additional, slower style checks that require a network connection.
          --new-cask        - Run various additional style checks to determine if a new cask is eligible
                              for Homebrew. This should be used when creating new casks and implies
                              `--strict` and `--online`.
          --download        - Audit the downloaded file
          --appcast         - Audit the appcast
          --token-conflicts - Audit for token conflicts

          Check <cask> for Homebrew coding style violations. This should be run before
          submitting a new cask. If no <casks> are provided, check all locally
          available casks. Will exit with a non-zero status if any errors are
          found, which can be useful for implementing pre-commit hooks.
        EOS
      end

      def self.help
        "verifies installability of Casks"
      end

      def run
        Homebrew.auditing = true
        strict = new_cask_arg? || strict_arg?
        token_conflicts = strict || token_conflicts_arg?

        online = new_cask_arg? || online_arg?
        download = online || download_arg?
        appcast = online || appcast_arg?

        failed_casks = casks(alternative: -> { Cask.to_a })
                       .reject do |cask|
          odebug "Auditing Cask #{cask}"
          result = Auditor.audit(cask, audit_download:        download,
                                       audit_appcast:         appcast,
                                       audit_online:          online,
                                       audit_strict:          strict,
                                       audit_new_cask:        new_cask_arg?,
                                       audit_token_conflicts: token_conflicts,
                                       quarantine:            quarantine?)

          result[:warnings].empty? && result[:errors].empty?
        end

        return if failed_casks.empty?

        raise CaskError, "audit failed for casks: #{failed_casks.join(" ")}"
      end
    end
  end
end

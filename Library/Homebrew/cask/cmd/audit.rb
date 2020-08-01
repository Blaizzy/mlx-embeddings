# frozen_string_literal: true

require "cli/parser"
require "cask/auditor"

module Cask
  class Cmd
    class Audit < AbstractCommand
      def self.description
        <<~EOS
          Check <cask> for Homebrew coding style violations. This should be run before
          submitting a new cask. If no <cask> is provided, checks all locally
          available casks. Will exit with a non-zero status if any errors are
          found, which can be useful for implementing pre-commit hooks.
        EOS
      end

      def self.parser
        super do
          switch "--download",
                 description: "Audit the downloaded file"
          switch "--appcast",
                 description: "Audit the appcast"
          switch "--token-conflicts",
                 description: "Audit for token conflicts"
          switch "--strict",
                 description: "Run additional, stricter style checks"
          switch "--online",
                 description: "Run additional, slower style checks that require a network connection"
          switch "--new-cask",
                 description: "Run various additional style checks to determine if a new cask is eligible
                               for Homebrew. This should be used when creating new casks and implies
                               `--strict` and `--online`"
        end
      end

      def run
        Homebrew.auditing = true
        strict = args.new_cask? || args.strict?
        online = args.new_cask? || args.online?

        options = {
          audit_download:        online || args.download?,
          audit_appcast:         online || args.appcast?,
          audit_online:          online,
          audit_strict:          strict,
          audit_new_cask:        args.new_cask?,
          audit_token_conflicts: strict || args.token_conflicts?,
          quarantine:            args.quarantine?,
          language:              args.language,
        }.compact

        options[:quarantine] = true if options[:quarantine].nil?

        failed_casks = casks(alternative: -> { Cask.to_a })
                       .reject do |cask|
          odebug "Auditing Cask #{cask}"
          result = Auditor.audit(cask, **options)

          result[:warnings].empty? && result[:errors].empty?
        end

        return if failed_casks.empty?

        raise CaskError, "audit failed for casks: #{failed_casks.join(" ")}"
      end
    end
  end
end

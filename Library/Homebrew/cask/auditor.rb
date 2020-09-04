# frozen_string_literal: true

require "cask/audit"

module Cask
  # Helper class for auditing all available languages of a cask.
  #
  # @api private
  class Auditor
    def self.audit(
      cask,
      audit_download: nil,
      audit_appcast: nil,
      audit_online: nil,
      audit_new_cask: nil,
      audit_strict: nil,
      audit_token_conflicts: nil,
      quarantine: nil,
      language: nil
    )
      new(
        cask,
        audit_download:        audit_download,
        audit_appcast:         audit_appcast,
        audit_online:          audit_online,
        audit_new_cask:        audit_new_cask,
        audit_strict:          audit_strict,
        audit_token_conflicts: audit_token_conflicts,
        quarantine:            quarantine,
        language:              language,
      ).audit
    end

    attr_reader :cask, :language

    def initialize(
      cask,
      audit_download: nil,
      audit_appcast: nil,
      audit_online: nil,
      audit_strict: nil,
      audit_token_conflicts: nil,
      audit_new_cask: nil,
      quarantine: nil,
      language: nil
    )
      @cask = cask
      @audit_download = audit_download
      @audit_appcast = audit_appcast
      @audit_online = audit_online
      @audit_new_cask = audit_new_cask
      @audit_strict = audit_strict
      @quarantine = quarantine
      @audit_token_conflicts = audit_token_conflicts
      @language = language
    end

    def audit
      warnings = Set.new
      errors = Set.new

      if !language && language_blocks
        language_blocks.each_key do |l|
          audit = audit_languages(l)
          puts audit.summary
          warnings += audit.warnings
          errors += audit.errors
        end
      else
        audit = audit_cask_instance(cask)
        puts audit.summary
        warnings += audit.warnings
        errors += audit.errors
      end

      { warnings: warnings, errors: errors }
    end

    private

    def audit_languages(languages)
      ohai "Auditing language: #{languages.map { |lang| "'#{lang}'" }.to_sentence}"
      localized_cask = CaskLoader.load(cask.sourcefile_path)
      config = localized_cask.config
      config.languages = languages
      localized_cask.config = config
      audit_cask_instance(localized_cask)
    end

    def audit_cask_instance(cask)
      audit = Audit.new(
        cask,
        appcast:         @audit_appcast,
        online:          @audit_online,
        strict:          @audit_strict,
        new_cask:        @audit_new_cask,
        token_conflicts: @audit_token_conflicts,
        download:        @audit_download,
        quarantine:      @quarantine,
      )
      audit.run!
      audit
    end

    def language_blocks
      cask.instance_variable_get(:@dsl).instance_variable_get(:@language_blocks)
    end
  end
end

# frozen_string_literal: true

module Cask
  class Auditor
    extend Predicable

    def self.audit(cask, audit_download: false, audit_appcast: false,
                   audit_online: false, audit_strict: false,
                   audit_token_conflicts: false, audit_new_cask: false,
                   quarantine: true, commit_range: nil)
      new(cask, audit_download: audit_download,
                audit_appcast: audit_appcast,
                audit_online: audit_online,
                audit_new_cask: audit_new_cask,
                audit_strict: audit_strict,
                audit_token_conflicts: audit_token_conflicts,
                quarantine: quarantine, commit_range: commit_range).audit
    end

    attr_reader :cask, :commit_range

    def initialize(cask, audit_download: false, audit_appcast: false,
                   audit_online: false, audit_strict: false,
                   audit_token_conflicts: false, audit_new_cask: false,
                   quarantine: true, commit_range: nil)
      @cask = cask
      @audit_download = audit_download
      @audit_appcast = audit_appcast
      @audit_online = audit_online
      @audit_strict = audit_strict
      @audit_new_cask = audit_new_cask
      @quarantine = quarantine
      @commit_range = commit_range
      @audit_token_conflicts = audit_token_conflicts
    end

    attr_predicate :audit_appcast?, :audit_download?, :audit_online?,
                   :audit_strict?, :audit_new_cask?, :audit_token_conflicts?, :quarantine?

    def audit
      warnings = Set.new
      errors = Set.new

      if !Homebrew.args.value("language") && language_blocks
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
      audit = Audit.new(cask, appcast:         audit_appcast?,
                              online:          audit_online?,
                              strict:          audit_strict?,
                              new_cask:        audit_new_cask?,
                              token_conflicts: audit_token_conflicts?,
                              download:        audit_download?,
                              quarantine:      quarantine?,
                              commit_range:    commit_range)
      audit.run!
      audit
    end

    def language_blocks
      cask.instance_variable_get(:@dsl).instance_variable_get(:@language_blocks)
    end
  end
end

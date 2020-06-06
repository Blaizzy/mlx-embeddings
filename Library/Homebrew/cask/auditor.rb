# frozen_string_literal: true

require "cask/download"

module Cask
  class Auditor
    include Checkable
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
      if !Homebrew.args.value("language") && language_blocks
        audit_all_languages
      else
        audit_cask_instance(cask)
      end
    end

    private

    def audit_all_languages
      saved_languages = MacOS.instance_variable_get(:@languages)
      begin
        language_blocks.keys.all?(&method(:audit_languages))
      ensure
        MacOS.instance_variable_set(:@languages, saved_languages)
      end
    end

    def audit_languages(languages)
      ohai "Auditing language: #{languages.map { |lang| "'#{lang}'" }.to_sentence}"
      MacOS.instance_variable_set(:@languages, languages)
      audit_cask_instance(CaskLoader.load(cask.sourcefile_path))
    end

    def audit_cask_instance(cask)
      download = audit_download? && Download.new(cask, quarantine: quarantine?)
      audit = Audit.new(cask, appcast:         audit_appcast?,
                              online:          audit_online?,
                              strict:          audit_strict?,
                              new_cask:        audit_new_cask?,
                              token_conflicts: audit_token_conflicts?,
                              download:        download,
                              commit_range:    commit_range)
      audit.run!
      puts audit.summary
      audit.success?
    end

    def language_blocks
      cask.instance_variable_get(:@dsl).instance_variable_get(:@language_blocks)
    end
  end
end

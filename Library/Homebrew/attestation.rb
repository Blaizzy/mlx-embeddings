# typed: true
# frozen_string_literal: true

require "date"
require "json"
require "utils/popen"
require "exceptions"

module Homebrew
  module Attestation
    HOMEBREW_CORE_REPO = "Homebrew/homebrew-core"
    HOMEBREW_CORE_CI_URI = "https://github.com/Homebrew/homebrew-core/.github/workflows/publish-commit-bottles.yml@refs/heads/master"

    BACKFILL_REPO = "trailofbits/homebrew-brew-verify"
    BACKFILL_REPO_CI_URI = "https://github.com/trailofbits/homebrew-brew-verify/.github/workflows/backfill_signatures.yml@refs/heads/main"

    # No backfill attestations after this date are considered valid.
    # @api private
    BACKFILL_CUTOFF = DateTime.new(2024, 3, 14).freeze

    # Verifies the given bottle against a cryptographic attestation of build provenance.
    #
    # The provenance is verified as originating from `signing_repo`, which is a `String`
    # that should be formatted as a GitHub `owner/repo`.
    #
    # Callers may additionally pass in `signing_workflow`, which will scope the attestation
    # down to an exact GitHub Actions workflow, in
    # `https://github/OWNER/REPO/.github/workflows/WORKFLOW.yml@REF` format.
    #
    # @return [Hash] the JSON-decoded response.
    # @raises [InvalidAttestationError] on any verification failures.
    #
    # @api private
    def self.check_attestation(bottle, signing_repo, signing_workflow = nil)
      cmd = [HOMEBREW_GH, "attestation", "verify", bottle.cached_download, "--repo", signing_repo, "--format", "json"]

      cmd += ["--cert-identity", signing_workflow] unless signing_workflow.nil?

      begin
        output = Utils.safe_popen_read(*cmd)
      rescue ErrorDuringExecution => e
        raise InvalidAttestationError, "attestation verification failed: #{e}"
      end

      begin
        data = JSON.parse(output)
      rescue JSON::ParserError
        raise InvalidAttestationError, "attestation verification returned malformed JSON"
      end

      raise InvalidAttestationError, "attestation output is empty" if data.empty?

      data
    end

    # Verifies the given bottle against a cryptographic attestation of build provenance
    # from homebrew-core's CI, falling back on a "backfill" attestation for older bottles.
    #
    # This is a specialization of `check_attestation` for homebrew-core.
    def self.check_core_attestation(bottle)
      begin
        attestation = check_attestation bottle, HOMEBREW_CORE_REPO
        return attestation
      rescue InvalidAttestationError
        odebug "falling back on backfilled attestation"
        backfill_attestation = check_attestation bottle, BACKFILL_REPO, BACKFILL_REPO_CI_URI
        timestamp = backfill_attestation.dig(0, "verificationResult", "verifiedTimestamps",
                                             0, "timestamp")

        raise InvalidAttestationError, "backfill attestation is missing verified timestamp" if timestamp.nil?

        if DateTime.parse(timestamp) > BACKFILL_CUTOFF
          raise InvalidAttestationError, "backfill attestation post-dates cutoff"
        end
      end

      backfill_attestation
    end
  end
end

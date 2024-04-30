# typed: strict
# frozen_string_literal: true

require "date"
require "json"
require "utils/popen"
require "exceptions"

module Homebrew
  module Attestation
    # @api private
    HOMEBREW_CORE_REPO = "Homebrew/homebrew-core"
    # @api private
    HOMEBREW_CORE_CI_URI = "https://github.com/Homebrew/homebrew-core/.github/workflows/publish-commit-bottles.yml@refs/heads/master"

    # @api private
    BACKFILL_REPO = "trailofbits/homebrew-brew-verify"

    # No backfill attestations after this date are considered valid.
    #
    # This date is shortly after the backfill operation for homebrew-core
    # completed, as can be seen here: <https://github.com/trailofbits/homebrew-brew-verify/attestations>.
    #
    # In effect, this means that, even if an attacker is able to compromise the backfill
    # signing workflow, they will be unable to convince a verifier to accept their newer,
    # malicious backfilled signatures.
    #
    # @api private
    BACKFILL_CUTOFF = T.let(DateTime.new(2024, 3, 14).freeze, DateTime)

    # Raised when attestation verification fails.
    #
    # @api private
    class InvalidAttestationError < RuntimeError; end

    # Returns a path to a suitable `gh` executable for attestation verification.
    #
    # @api private
    sig { returns(Pathname) }
    def self.gh_executable
      # NOTE: We disable HOMEBREW_VERIFY_ATTESTATIONS when installing `gh` itself,
      # to prevent a cycle during bootstrapping. This can eventually be resolved
      # by vendoring a pure-Ruby Sigstore verifier client.
      @gh_executable ||= T.let(with_env("HOMEBREW_VERIFY_ATTESTATIONS" => nil) do
        ensure_executable!("gh")
      end, T.nilable(Pathname))
    end

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
    # @raise [InvalidAttestationError] on any verification failures
    #
    # @api private
    sig {
      params(bottle: Bottle, signing_repo: String,
             signing_workflow: T.nilable(String), subject: T.nilable(String)).returns(T::Hash[T.untyped, T.untyped])
    }
    def self.check_attestation(bottle, signing_repo, signing_workflow = nil, subject = nil)
      cmd = [gh_executable, "attestation", "verify", bottle.cached_download, "--repo", signing_repo, "--format",
             "json"]

      cmd += ["--cert-identity", signing_workflow] if signing_workflow.present?

      begin
        output = Utils.safe_popen_read(*cmd)
      rescue ErrorDuringExecution => e
        raise InvalidAttestationError, "attestation verification failed: #{e}"
      end

      begin
        attestations = JSON.parse(output)
      rescue JSON::ParserError
        raise InvalidAttestationError, "attestation verification returned malformed JSON"
      end

      # `gh attestation verify` returns a JSON array of one or more results,
      # for all attestations that match the input's digest. We want to additionally
      # filter these down to just the attestation whose subject matches the bottle's name.
      subject = bottle.filename.to_s if subject.blank?
      attestation = attestations.find do |a|
        a.dig("verificationResult", "statement", "subject", 0, "name") == subject
      end

      raise InvalidAttestationError, "no attestation matches subject" if attestation.blank?

      attestation
    end

    # Verifies the given bottle against a cryptographic attestation of build provenance
    # from homebrew-core's CI, falling back on a "backfill" attestation for older bottles.
    #
    # This is a specialization of `check_attestation` for homebrew-core.
    #
    # @return [Hash] the JSON-decoded response
    # @raise [InvalidAttestationError] on any verification failures
    #
    # @api private
    sig { params(bottle: Bottle).returns(T::Hash[T.untyped, T.untyped]) }
    def self.check_core_attestation(bottle)
      begin
        attestation = check_attestation bottle, HOMEBREW_CORE_REPO, HOMEBREW_CORE_CI_URI
        return attestation
      rescue InvalidAttestationError
        odebug "falling back on backfilled attestation for #{bottle}"

        # Our backfilled attestation is a little unique: the subject is not just the bottle
        # filename, but also has the bottle's hosted URL hash prepended to it.
        # This was originally unintentional, but has a virtuous side effect of further
        # limiting domain separation on the backfilled signatures (by committing them to
        # their original bottle URLs).
        url_sha256 = Digest::SHA256.hexdigest(bottle.url)
        subject = "#{url_sha256}--#{bottle.filename}"

        # We don't pass in a signing workflow for backfill signatures because
        # some backfilled bottle signatures were signed from the 'backfill'
        # branch, and others from 'main' of trailofbits/homebrew-brew-verify
        # so the signing workflow is slightly different which causes some bottles to incorrectly
        # fail when checking their attestation. This shouldn't meaningfully affect security
        # because if somehow someone could generate false backfill attestations
        # from a different workflow we will still catch it because the
        # attestation would have been generated after our cutoff date.
        backfill_attestation = check_attestation bottle, BACKFILL_REPO, nil, subject
        timestamp = backfill_attestation.dig("verificationResult", "verifiedTimestamps",
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

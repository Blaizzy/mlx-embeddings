# typed: true
# frozen_string_literal: true

require "cxxstdlib"
require "json"
require "development_tools"
require "extend/cachable"
require "utils/curl"

# Rather than calling `new` directly, use one of the class methods like {SBOM.create}.
class SBOM
  extend Cachable

  FILENAME = "sbom.spdx.json"
  SCHEMA_URL = "https://spdx.github.io/spdx-3-model/model.jsonld"
  SCHEMA_FILENAME = "sbom.spdx.schema.3.json"
  SCHEMA_CACHE_TARGET = (HOMEBREW_CACHE/"sbom/#{SCHEMA_FILENAME}").freeze

  # Instantiates a {SBOM} for a new installation of a formula.
  sig { params(formula: Formula, tab: Tab).returns(T.attached_class) }
  def self.create(formula, tab)
    attributes = {
      name:                 formula.name,
      homebrew_version:     HOMEBREW_VERSION,
      spdxfile:             SBOM.spdxfile(formula),
      time:                 Time.now.to_i,
      source_modified_time: tab.source_modified_time.to_i,
      compiler:             tab.compiler,
      stdlib:               tab.stdlib,
      runtime_dependencies: SBOM.runtime_deps_hash(Array(tab.runtime_dependencies)),
      license:              SPDX.license_expression_to_string(formula.license),
      built_on:             DevelopmentTools.build_system_info,
      source:               {
        path:         formula.specified_path.to_s,
        tap:          formula.tap&.name,
        tap_git_head: nil, # Filled in later if possible
        spec:         formula.active_spec_sym.to_s,
        patches:      formula.stable&.patches,
        bottle:       formula.bottle_hash,
        stable:       {
          version:  formula.stable&.version,
          url:      formula.stable&.url,
          checksum: formula.stable&.checksum,
        },
      },
    }

    # We can only get `tap_git_head` if the tap is installed locally
    attributes[:source][:tap_git_head] = T.must(formula.tap).git_head if formula.tap&.installed?

    new(attributes)
  end

  sig { params(formula: Formula).returns(Pathname) }
  def self.spdxfile(formula)
    formula.prefix/FILENAME
  end

  sig { params(deps: T::Array[T::Hash[String, String]]).returns(T::Array[T::Hash[String, String]]) }
  def self.runtime_deps_hash(deps)
    deps.map do |dep|
      full_name = dep.fetch("full_name")
      dep_formula = Formula[full_name]
      {
        "full_name"           => full_name,
        "pkg_version"         => dep.fetch("pkg_version"),
        "name"                => dep_formula.name,
        "license"             => SPDX.license_expression_to_string(dep_formula.license),
        "bottle"              => dep_formula.bottle_hash,
        "formula_pkg_version" => dep_formula.pkg_version.to_s,
      }
    end
  end

  sig { params(formula: Formula).returns(T::Boolean) }
  def self.exist?(formula)
    spdxfile(formula).exist?
  end

  sig { returns(T::Hash[String, String]) }
  def self.fetch_schema!
    return @schema if @schema.present?

    url = SCHEMA_URL
    target = SCHEMA_CACHE_TARGET
    quieter = target.exist? && !target.empty?

    curl_args = Utils::Curl.curl_args(retries: 0)
    curl_args += ["--silent", "--time-cond", target.to_s] if quieter

    begin
      unless quieter
        oh1 "Fetching SBOM schema"
        ohai "Downloading #{url}"
      end
      Utils::Curl.curl_download(*curl_args, url, to: target, retries: 0)
      FileUtils.touch(target, mtime: Time.now)
    rescue ErrorDuringExecution
      target.unlink if target.exist? && target.empty?

      if target.exist?
        opoo "SBOM schema update failed, falling back to cached version."
      else
        opoo "Failed to fetch SBOM schema, cannot perform SBOM validation!"

        return {}
      end
    end

    @schema = begin
      JSON.parse(target.read, freeze: true)
    rescue JSON::ParserError
      target.unlink
      opoo "Failed to fetch SBOM schema, cached version corrupted, cannot perform SBOM validation!"
      {}
    end
  end

  sig { returns(T::Boolean) }
  def valid?
    unless require? "json_schemer"
      error_message = "Need json_schemer to validate SBOM, run `brew install-bundler-gems --add-groups=bottle`!"
      odie error_message if ENV["HOMEBREW_ENFORCE_SBOM"]
      return false
    end

    schema = SBOM.fetch_schema!
    if schema.blank?
      error_message = "Could not fetch JSON schema to validate SBOM!"
      ENV["HOMEBREW_ENFORCE_SBOM"] ? odie(error_message) : opoo(error_message)
      return false
    end

    schemer = JSONSchemer.schema(schema)
    data = to_spdx_sbom
    return true if schemer.valid?(data)

    opoo "SBOM validation errors:"
    schemer.validate(data).to_a.each do |error|
      puts error["error"]
    end

    odie "Failed to validate SBOM against JSON schema!" if ENV["HOMEBREW_ENFORCE_SBOM"]

    false
  end

  sig { params(validate: T::Boolean).void }
  def write(validate: true)
    # If this is a new installation, the cache of installed formulae
    # will no longer be valid.
    Formula.clear_cache unless spdxfile.exist?

    self.class.cache[spdxfile] = self

    if validate && !valid?
      opoo "SBOM is not valid, not writing to disk!"
      return
    end

    spdxfile.atomic_write(JSON.pretty_generate(to_spdx_sbom))
  end

  private

  attr_reader :name, :homebrew_version, :time, :stdlib, :source, :built_on, :license
  attr_accessor :spdxfile

  sig { params(attributes: Hash).void }
  def initialize(attributes = {})
    attributes.each { |key, value| instance_variable_set(:"@#{key}", value) }
  end

  sig { params(runtime_dependency_declaration: T::Array[Hash], compiler_declaration: Hash).returns(T::Array[Hash]) }
  def generate_relations_json(runtime_dependency_declaration, compiler_declaration)
    runtime = runtime_dependency_declaration.map do |dependency|
      {
        spdxElementId:      dependency[:SPDXID],
        relationshipType:   "RUNTIME_DEPENDENCY_OF",
        relatedSpdxElement: "SPDXRef-Bottle-#{name}",
      }
    end
    patches = source[:patches].each_with_index.map do |_patch, index|
      {
        spdxElementId:      "SPDXRef-Patch-#{name}-#{index}",
        relationshipType:   "PATCH_APPLIED",
        relatedSpdxElement: "SPDXRef-Archive-#{name}-src",
      }
    end

    base = [
      {
        spdxElementId:      "SPDXRef-File-#{name}",
        relationshipType:   "PACKAGE_OF",
        relatedSpdxElement: "SPDXRef-Archive-#{name}-src",
      },
      {
        spdxElementId:      "SPDXRef-Compiler",
        relationshipType:   "BUILD_TOOL_OF",
        relatedSpdxElement: "SPDXRef-Package-#{name}-src",
      },
    ]

    if compiler_declaration["SPDXRef-Stdlib"].present?
      base << {
        spdxElementId:      "SPDXRef-Stdlib",
        relationshipType:   "DEPENDENCY_OF",
        relatedSpdxElement: "SPDXRef-Bottle-#{name}",
      }
    end

    runtime + patches + base
  end

  sig {
    params(runtime_dependency_declaration: T::Array[Hash],
           compiler_declaration:           Hash).returns(T::Array[T::Hash[Symbol,
                                                                          T.any(String,
                                                                                T::Array[T::Hash[Symbol, String]])]])
  }
  def generate_packages_json(runtime_dependency_declaration, compiler_declaration)
    bottle = []
    if (bottle_info = get_bottle_info(source[:bottle]))
      bottle << {
        SPDXID:           "SPDXRef-Bottle-#{name}",
        name:             name.to_s,
        versionInfo:      stable_version.to_s,
        filesAnalyzed:    false,
        licenseDeclared:  assert_value(nil),
        builtDate:        source_modified_time.to_s,
        licenseConcluded: license,
        downloadLocation: bottle_info.fetch("url"),
        copyrightText:    assert_value(nil),
        externalRefs:     [
          {
            referenceCategory: "PACKAGE-MANAGER",
            referenceLocator:  "pkg:brew/#{tap}/#{name}@#{stable_version}",
            referenceType:     "purl",
          },
        ],
        checksums:        [
          {
            algorithm:     "SHA256",
            checksumValue: bottle_info.fetch("sha256"),
          },
        ],
      }
    end

    [
      {
        SPDXID:           "SPDXRef-Archive-#{name}-src",
        name:             name.to_s,
        versionInfo:      stable_version.to_s,
        filesAnalyzed:    false,
        licenseDeclared:  assert_value(nil),
        builtDate:        source_modified_time.to_s,
        licenseConcluded: assert_value(license),
        downloadLocation: source[:stable][:url],
        copyrightText:    assert_value(nil),
        externalRefs:     [],
        checksums:        [
          {
            algorithm:     "SHA256",
            checksumValue: source[:stable][:checksum].to_s,
          },
        ],
      },
    ] + runtime_dependency_declaration + compiler_declaration.values + bottle
  end

  sig { returns(T::Array[T::Hash[Symbol, T.any(T::Boolean, String, T::Array[T::Hash[Symbol, String]])]]) }
  def full_spdx_runtime_dependencies
    return [] unless @runtime_dependencies.present?

    @runtime_dependencies.compact.filter_map do |dependency|
      next unless dependency.present?

      bottle_info = get_bottle_info(dependency["bottle"])
      next unless bottle_info.present?

      # Only set bottle URL if the dependency is the same version as the formula/bottle.
      bottle_url = bottle_info["url"] if dependency["pkg_version"] == dependency["formula_pkg_version"]

      {
        SPDXID:           "SPDXRef-Package-SPDXRef-#{dependency["name"].tr("/", "-")}-#{dependency["pkg_version"]}",
        name:             dependency["name"],
        versionInfo:      dependency["pkg_version"],
        filesAnalyzed:    false,
        licenseDeclared:  assert_value(nil),
        licenseConcluded: assert_value(dependency["license"]),
        downloadLocation: assert_value(bottle_url),
        copyrightText:    assert_value(nil),
        checksums:        [
          {
            algorithm:     "SHA256",
            checksumValue: assert_value(bottle_info["sha256"]),
          },
        ],
        externalRefs:     [
          {
            referenceCategory: "PACKAGE-MANAGER",
            referenceLocator:  "pkg:brew/#{dependency["full_name"]}@#{dependency["pkg_version"]}",
            referenceType:     :purl,
          },
        ],
      }
    end
  end

  sig { returns(T::Hash[Symbol, T.any(String, T::Array[T::Hash[Symbol, String]])]) }
  def to_spdx_sbom
    runtime_full = full_spdx_runtime_dependencies

    compiler_info = {
      "SPDXRef-Compiler" => {
        SPDXID:           "SPDXRef-Compiler",
        name:             compiler.to_s,
        versionInfo:      assert_value(built_on["xcode"]),
        filesAnalyzed:    false,
        licenseDeclared:  assert_value(nil),
        licenseConcluded: assert_value(nil),
        copyrightText:    assert_value(nil),
        downloadLocation: assert_value(nil),
        checksums:        [],
        externalRefs:     [],
      },
    }

    if stdlib.present?
      compiler_info["SPDXRef-Stdlib"] = {
        SPDXID:           "SPDXRef-Stdlib",
        name:             stdlib,
        versionInfo:      stdlib,
        filesAnalyzed:    false,
        licenseDeclared:  assert_value(nil),
        licenseConcluded: assert_value(nil),
        copyrightText:    assert_value(nil),
        downloadLocation: assert_value(nil),
        checksums:        [],
        externalRefs:     [],
      }
    end

    packages = generate_packages_json(runtime_full, compiler_info)
    {
      SPDXID:            "SPDXRef-DOCUMENT",
      spdxVersion:       "SPDX-2.3",
      name:              "SBOM-SPDX-#{name}-#{stable_version}",
      creationInfo:      {
        created:  DateTime.now.to_s,
        creators: ["Tool: https://github.com/homebrew/brew@#{homebrew_version}"],
      },
      dataLicense:       "CC0-1.0",
      documentNamespace: "https://formulae.brew.sh/spdx/#{name}-#{stable_version}.json",
      documentDescribes: packages.map { |dependency| dependency[:SPDXID] },
      files:             [],
      packages:,
      relationships:     generate_relations_json(runtime_full, compiler_info),
    }
  end

  sig { params(base: T.nilable(T::Hash[String, Hash])).returns(T.nilable(T::Hash[String, String])) }
  def get_bottle_info(base)
    return unless base.present?

    files = base["files"].presence
    return unless files

    files[Utils::Bottles.tag.to_sym] || files[:all]
  end

  sig { returns(Symbol) }
  def compiler
    @compiler.presence&.to_sym || DevelopmentTools.default_compiler
  end

  sig { returns(T.nilable(Tap)) }
  def tap
    tap_name = source[:tap]
    Tap.fetch(tap_name) if tap_name
  end

  sig { returns(T.nilable(Version)) }
  def stable_version
    source[:stable][:version]
  end

  sig { returns(Time) }
  def source_modified_time
    Time.at(@source_modified_time || 0).utc
  end

  sig { params(val: T.untyped).returns(T.any(String, Symbol)) }
  def assert_value(val)
    return :NOASSERTION.to_s unless val.present?

    val
  end
end

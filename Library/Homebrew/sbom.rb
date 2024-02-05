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
  SCHEMA = "https://raw.githubusercontent.com/spdx/spdx-spec/v2.3/schemas/spdx-schema.json"

  attr_accessor :homebrew_version, :spdxfile, :built_as_bottle, :installed_as_dependency, :installed_on_request,
                :changed_files, :poured_from_bottle, :loaded_from_api, :time, :stdlib, :aliases, :arch, :source,
                :built_on, :license, :name
  attr_writer :compiler, :runtime_dependencies, :source_modified_time

  # Instantiates a {SBOM} for a new installation of a formula.
  sig { params(formula: Formula, compiler: T.nilable(String), stdlib: T.nilable(String)).returns(T.attached_class) }
  def self.create(formula, compiler: nil, stdlib: nil)
    runtime_deps = formula.runtime_formula_dependencies(undeclared: false)

    attributes = {
      name:                    formula.name,
      homebrew_version:        HOMEBREW_VERSION,
      spdxfile:                formula.prefix/FILENAME,
      built_as_bottle:         formula.build.bottle?,
      installed_as_dependency: false,
      installed_on_request:    false,
      poured_from_bottle:      false,
      loaded_from_api:         false,
      time:                    Time.now.to_i,
      source_modified_time:    formula.source_modified_time.to_i,
      compiler:,
      stdlib:,
      aliases:                 formula.aliases,
      runtime_dependencies:    SBOM.runtime_deps_hash(runtime_deps),
      arch:                    Hardware::CPU.arch,
      license:                 SPDX.license_expression_to_string(formula.license),
      built_on:                DevelopmentTools.build_system_info,
      source:                  {
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

  sig { params(attributes: Hash).void }
  def initialize(attributes = {})
    attributes.each { |key, value| instance_variable_set(:"@#{key}", value) }
  end

  sig { returns(T::Boolean) }
  def valid?
    data = to_spdx_sbom

    schema_string, _, status = Utils::Curl.curl_output(SCHEMA)

    opoo "Failed to fetch schema!" unless status.success?

    require "json_schemer"

    schemer = JSONSchemer.schema(schema_string)

    return true if schemer.valid?(data)

    opoo "SBOM validation errors:"
    schemer.validate(data).to_a.each do |error|
      ohai error["error"]
    end

    odie "Failed to validate SBOM agains schema!" if ENV["HOMEBREW_ENFORCE_SBOM"]

    false
  end

  sig { void }
  def write
    # If this is a new installation, the cache of installed formulae
    # will no longer be valid.
    Formula.clear_cache unless spdxfile.exist?

    self.class.cache[spdxfile] = self

    unless valid?
      opoo "SBOM is not valid, not writing to disk!"
      return
    end

    spdxfile.atomic_write(JSON.pretty_generate(to_spdx_sbom))
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
      base += {
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
    if get_bottle_info(source[:bottle])
      bottle << {
        SPDXID:           "SPDXRef-Bottle-#{name}",
        name:             name.to_s,
        versionInfo:      stable_version.to_s,
        filesAnalyzed:    false,
        licenseDeclared:  assert_value(nil),
        builtDate:        source_modified_time.to_s,
        licenseConcluded: license,
        downloadLocation: T.must(get_bottle_info(source[:bottle]))["url"],
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
            checksumValue: T.must(get_bottle_info(source[:bottle]))["sha256"],
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

      {
        SPDXID:           "SPDXRef-Package-SPDXRef-#{dependency["name"].tr("/", "-")}-#{dependency["version"]}",
        name:             dependency["name"],
        versionInfo:      dependency["pkg_version"],
        filesAnalyzed:    false,
        licenseDeclared:  assert_value(nil),
        licenseConcluded: assert_value(dependency["license"]),
        downloadLocation: assert_value(bottle_info.present? ? bottle_info["url"] : nil),
        copyrightText:    assert_value(nil),
        checksums:        [
          {
            algorithm:     "SHA256",
            checksumValue: assert_value(bottle_info.present? ? bottle_info["sha256"] : nil),
          },
        ],
        externalRefs:     [
          {
            referenceCategory: "PACKAGE-MANAGER",
            referenceLocator:  "pkg:brew/#{dependency["full_name"]}@#{dependency["version"]}",
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

  sig { params(deps: T::Array[Formula]).returns(T::Array[T::Hash[Symbol, String]]) }
  def self.runtime_deps_hash(deps)
    deps.map do |dep|
      {
        full_name:         dep.full_name,
        name:              dep.name,
        version:           dep.version.to_s,
        revision:          dep.revision,
        pkg_version:       dep.pkg_version.to_s,
        declared_directly: true,
        license:           SPDX.license_expression_to_string(dep.license),
        bottle:            dep.bottle_hash,
      }
    end
  end

  private

  sig { params(base: T.nilable(T::Hash[String, Hash])).returns(T.nilable(T::Hash[String, String])) }
  def get_bottle_info(base)
    return unless base.present?
    return unless base.key?("files")

    T.must(base["files"])[Utils::Bottles.tag.to_sym]
  end

  sig { returns(T::Boolean) }
  def stable?
    spec == :stable
  end

  sig { returns(Symbol) }
  def compiler
    @compiler || DevelopmentTools.default_compiler
  end

  sig { returns(CxxStdlib) }
  def cxxstdlib
    # Older sboms won't have these values, so provide sensible defaults
    lib = stdlib.to_sym if stdlib
    CxxStdlib.create(lib, compiler.to_sym)
  end

  sig { returns(T::Boolean) }
  def built_bottle?
    built_as_bottle && !poured_from_bottle
  end

  sig { returns(T::Boolean) }
  def bottle?
    built_as_bottle
  end

  sig { returns(T.nilable(Tap)) }
  def tap
    tap_name = source[:tap]
    Tap.fetch(tap_name) if tap_name
  end

  sig { returns(Symbol) }
  def spec
    source[:spec].to_sym
  end

  sig { returns(T.nilable(Version)) }
  def stable_version
    source[:stable][:version]
  end

  sig { returns(Time) }
  def source_modified_time
    Time.at(@source_modified_time || 0)
  end

  sig { params(val: T.untyped).returns(T.any(String, Symbol)) }
  def assert_value(val)
    return :NOASSERTION.to_s unless val.present?

    val
  end
end

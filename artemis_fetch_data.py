from uuid import UUID

from artemis_tools.clients.falcon.client import FalconClient, FalconSettings

METRICS = ['cpu', 'memory', "runtime"]

def falcon_client() -> FalconClient:
    falcon_settings = FalconSettings.with_env_prefix(
        "falcon",
        _env_file="../.env",
    )
    return FalconClient(falcon_settings)


def get_spec_info(all_constructs, construct_id, spec_id):
    for s in all_constructs[construct_id].custom_specs:
        if s.id == spec_id:
            return s

if __name__ == '__main__':

    OPTIMISATION_ID = "dc5076f4-03be-4fc8-bfb0-80b432fa8a84"
    construct_id = UUID("c9d0f1d6-4b37-48b3-808e-200d9b2b2597")
    spec_id = UUID("6fc1d363-606f-422b-b3af-cbbb2134921b")

    f = falcon_client()
    f.authenticate()
    opt = f.get_optimisation(OPTIMISATION_ID)
    project_id = opt.project_id
    res = f.get("/code/optimisations/{}/solutions?perPage=-1".format(OPTIMISATION_ID))
    all_constructs = f.get_constructs_info(project_id)
    print(all_constructs)


    spec = get_spec_info(all_constructs, construct_id, spec_id)
    print(spec)
    print(spec.content)



    # # print(res["docs"])
    # for sol in res["docs"]:
    #     print(sol)
    #     if sol["status"] != "failed" and sol["status"] != "running":
    #         print(f"Solution {sol['id']}")
    #         # print(sol)
    #         for m in METRICS:
    #             print(f"\t {m}: {sol['results']['values'][m]}")